// Copyright 2026 The Cloud Hypervisor Authors. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom};
use std::ops::Range;
use std::os::fd::AsRawFd;
use std::os::unix::fs::FileExt;
use std::os::unix::fs::MetadataExt;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use vmm_sys_util::eventfd::EventFd;

use crate::async_io::{AsyncIo, AsyncIoError, AsyncIoResult, BorrowedDiskFd, DiskFileError};
use crate::disk_file;
use crate::error::{BlockError, BlockErrorKind, BlockResult, ErrorOp};
use crate::query_device_size;

const VMDK_SPARSE_MAGIC: u32 = 0x564d_444b;
const VMDK_SECTOR_SIZE: u64 = 512;
const VMDK_MAX_DESCRIPTOR_LEN: u64 = 2 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum VmdkAccess {
    ReadWrite,
    ReadOnly,
    NoAccess,
}

#[derive(Debug)]
enum ParsedStorage {
    Flat { filename: String, offset: u64 },
    Zero,
}

#[derive(Debug)]
struct ParsedExtent {
    access: VmdkAccess,
    sectors: u64,
    storage: Option<ParsedStorage>,
}

#[derive(Debug)]
enum ExtentStorage {
    Flat { file: File, offset: u64 },
    Zero,
}

#[derive(Debug)]
struct VmdkExtent {
    access: VmdkAccess,
    disk_range: Range<u64>,
    storage: Option<ExtentStorage>,
}

#[derive(Debug)]
struct VmdkDisk {
    descriptor_file: File,
    virtual_size: u64,
    extents: Vec<VmdkExtent>,
}

impl VmdkDisk {
    fn read_into_slice(&self, mut disk_offset: u64, mut dst: &mut [u8]) -> io::Result<usize> {
        if disk_offset >= self.virtual_size {
            return Ok(0);
        }
        if disk_offset
            .checked_add(dst.len() as u64)
            .map(|end| end > self.virtual_size)
            .unwrap_or(true)
        {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "read exceeds VMDK virtual size",
            ));
        }

        let mut total = 0usize;
        while !dst.is_empty() {
            let extent = self
                .extent_for_offset(disk_offset)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing VMDK extent"))?;
            let in_extent = disk_offset - extent.disk_range.start;
            let chunk = (extent.disk_range.end - disk_offset).min(dst.len() as u64) as usize;
            let (head, tail) = dst.split_at_mut(chunk);
            self.read_extent(extent, in_extent, head)?;
            disk_offset += chunk as u64;
            dst = tail;
            total += chunk;
        }

        Ok(total)
    }

    fn write_from_slice(&self, mut disk_offset: u64, mut src: &[u8]) -> io::Result<usize> {
        if disk_offset >= self.virtual_size {
            return Ok(0);
        }
        if disk_offset
            .checked_add(src.len() as u64)
            .map(|end| end > self.virtual_size)
            .unwrap_or(true)
        {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "write exceeds VMDK virtual size",
            ));
        }

        let mut total = 0usize;
        while !src.is_empty() {
            let extent = self
                .extent_for_offset(disk_offset)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing VMDK extent"))?;
            let in_extent = disk_offset - extent.disk_range.start;
            let chunk = (extent.disk_range.end - disk_offset).min(src.len() as u64) as usize;
            let (head, tail) = src.split_at(chunk);
            self.write_extent(extent, in_extent, head)?;
            disk_offset += chunk as u64;
            src = tail;
            total += chunk;
        }

        Ok(total)
    }

    fn extent_for_offset(&self, offset: u64) -> Option<&VmdkExtent> {
        self.extents
            .binary_search_by(|extent| {
                if extent.disk_range.contains(&offset) {
                    std::cmp::Ordering::Equal
                } else if extent.disk_range.end <= offset {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Greater
                }
            })
            .ok()
            .map(|index| &self.extents[index])
    }

    fn read_extent(&self, extent: &VmdkExtent, in_extent: u64, dst: &mut [u8]) -> io::Result<()> {
        if extent.access == VmdkAccess::NoAccess {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "attempted to read a NOACCESS extent",
            ));
        }

        match extent.storage.as_ref() {
            Some(ExtentStorage::Flat { file, offset }) => {
                let file_offset = offset
                    .checked_add(in_extent)
                    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "extent overflow"))?;
                let count = file.read_at(dst, file_offset)?;
                if count != dst.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "short read from VMDK extent",
                    ));
                }
                Ok(())
            }
            Some(ExtentStorage::Zero) => {
                dst.fill(0);
                Ok(())
            }
            None => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "extent data source missing",
            )),
        }
    }

    fn write_extent(&self, extent: &VmdkExtent, in_extent: u64, src: &[u8]) -> io::Result<()> {
        match extent.access {
            VmdkAccess::ReadWrite => {}
            VmdkAccess::ReadOnly | VmdkAccess::NoAccess => {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "attempted to write a read-only VMDK extent",
                ));
            }
        }

        match extent.storage.as_ref() {
            Some(ExtentStorage::Flat { file, offset }) => {
                let file_offset = offset
                    .checked_add(in_extent)
                    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "extent overflow"))?;
                let count = file.write_at(src, file_offset)?;
                if count != src.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "short write to VMDK extent",
                    ));
                }
                Ok(())
            }
            Some(ExtentStorage::Zero) => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "write to ZERO extent is not supported",
            )),
            None => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "extent data source missing",
            )),
        }
    }

    fn fsync_all_data_files(&self) -> io::Result<()> {
        for extent in &self.extents {
            if let Some(ExtentStorage::Flat { file, .. }) = extent.storage.as_ref() {
                file.sync_all()?;
            }
        }
        Ok(())
    }

    fn physical_size(&self) -> io::Result<u64> {
        let mut seen_files = HashSet::new();
        let descriptor_md = self.descriptor_file.metadata()?;
        seen_files.insert((descriptor_md.dev(), descriptor_md.ino()));
        let mut size = query_device_size(&self.descriptor_file)?.1;

        for extent in &self.extents {
            if let Some(ExtentStorage::Flat { file, .. }) = extent.storage.as_ref() {
                let md = file.metadata()?;
                let key = (md.dev(), md.ino());
                if seen_files.insert(key) {
                    size = size
                        .checked_add(query_device_size(file)?.1)
                        .ok_or_else(|| {
                            io::Error::new(io::ErrorKind::InvalidData, "size overflow")
                        })?;
                }
            }
        }
        Ok(size)
    }
}

#[derive(Debug)]
pub struct VmdkDiskSync {
    disk: Arc<Mutex<VmdkDisk>>,
}

impl VmdkDiskSync {
    pub fn new(
        mut descriptor_file: File,
        descriptor_path: &Path,
        readonly: bool,
        direct: bool,
    ) -> BlockResult<Self> {
        let descriptor = read_descriptor(&mut descriptor_file)
            .map_err(|e| classify_open_error(e).with_op(ErrorOp::Open))?;
        let parsed = parse_descriptor(&descriptor)
            .map_err(|e| classify_open_error(e).with_op(ErrorOp::Open))?;
        let extents = open_extents(descriptor_path, parsed, readonly, direct)
            .map_err(|e| classify_open_error(e).with_op(ErrorOp::Open))?;
        let virtual_size = extents.last().map_or(0, |e| e.disk_range.end);

        Ok(Self {
            disk: Arc::new(Mutex::new(VmdkDisk {
                descriptor_file,
                virtual_size,
                extents,
            })),
        })
    }
}

impl disk_file::DiskSize for VmdkDiskSync {
    fn logical_size(&self) -> BlockResult<u64> {
        Ok(self.disk.lock().unwrap().virtual_size)
    }
}

impl disk_file::PhysicalSize for VmdkDiskSync {
    fn physical_size(&self) -> BlockResult<u64> {
        self.disk
            .lock()
            .unwrap()
            .physical_size()
            .map_err(|e| BlockError::new(BlockErrorKind::Io, e))
    }
}

impl disk_file::DiskFd for VmdkDiskSync {
    fn fd(&self) -> BorrowedDiskFd<'_> {
        BorrowedDiskFd::new(self.disk.lock().unwrap().descriptor_file.as_raw_fd())
    }
}

impl disk_file::Geometry for VmdkDiskSync {}

impl disk_file::SparseCapable for VmdkDiskSync {}

impl disk_file::Resizable for VmdkDiskSync {
    fn resize(&mut self, _size: u64) -> BlockResult<()> {
        Err(BlockError::new(
            BlockErrorKind::UnsupportedFeature,
            DiskFileError::ResizeError(io::Error::other("resize not supported for VMDK")),
        )
        .with_op(ErrorOp::Resize))
    }
}

impl disk_file::DiskFile for VmdkDiskSync {}

impl disk_file::AsyncDiskFile for VmdkDiskSync {
    fn try_clone(&self) -> BlockResult<Box<dyn disk_file::AsyncDiskFile>> {
        Ok(Box::new(Self {
            disk: Arc::clone(&self.disk),
        }))
    }

    fn create_async_io(&self, _ring_depth: u32) -> BlockResult<Box<dyn AsyncIo>> {
        Ok(Box::new(VmdkSync::new(Arc::clone(&self.disk))))
    }
}

pub struct VmdkSync {
    disk: Arc<Mutex<VmdkDisk>>,
    eventfd: EventFd,
    completion_list: VecDeque<(u64, i32)>,
}

impl VmdkSync {
    pub fn new(disk: Arc<Mutex<VmdkDisk>>) -> Self {
        Self {
            disk,
            eventfd: EventFd::new(libc::EFD_NONBLOCK).expect("Failed creating EventFd for VMDK"),
            completion_list: VecDeque::new(),
        }
    }
}

impl AsyncIo for VmdkSync {
    fn notifier(&self) -> &EventFd {
        &self.eventfd
    }

    fn read_vectored(
        &mut self,
        offset: libc::off_t,
        iovecs: &[libc::iovec],
        user_data: u64,
    ) -> AsyncIoResult<()> {
        if offset < 0 {
            return Err(AsyncIoError::ReadVectored(io::Error::new(
                io::ErrorKind::InvalidInput,
                "negative read offset",
            )));
        }

        let mut pos = offset as u64;
        let mut total = 0usize;
        for iovec in iovecs {
            // SAFETY: `iovec` comes from the virtio request parser and is valid
            // for reads for the duration of this operation.
            let dst = unsafe {
                std::slice::from_raw_parts_mut(iovec.iov_base.cast::<u8>(), iovec.iov_len)
            };
            let read = self
                .disk
                .lock()
                .unwrap()
                .read_into_slice(pos, dst)
                .map_err(AsyncIoError::ReadVectored)?;
            pos += read as u64;
            total += read;
        }

        self.completion_list.push_back((user_data, total as i32));
        self.eventfd.write(1).unwrap();
        Ok(())
    }

    fn write_vectored(
        &mut self,
        offset: libc::off_t,
        iovecs: &[libc::iovec],
        user_data: u64,
    ) -> AsyncIoResult<()> {
        if offset < 0 {
            return Err(AsyncIoError::WriteVectored(io::Error::new(
                io::ErrorKind::InvalidInput,
                "negative write offset",
            )));
        }

        let mut pos = offset as u64;
        let mut total = 0usize;
        for iovec in iovecs {
            // SAFETY: `iovec` comes from the virtio request parser and is valid
            // for writes for the duration of this operation.
            let src =
                unsafe { std::slice::from_raw_parts(iovec.iov_base.cast::<u8>(), iovec.iov_len) };
            let written = self
                .disk
                .lock()
                .unwrap()
                .write_from_slice(pos, src)
                .map_err(AsyncIoError::WriteVectored)?;
            pos += written as u64;
            total += written;
        }

        self.completion_list.push_back((user_data, total as i32));
        self.eventfd.write(1).unwrap();
        Ok(())
    }

    fn fsync(&mut self, user_data: Option<u64>) -> AsyncIoResult<()> {
        self.disk
            .lock()
            .unwrap()
            .fsync_all_data_files()
            .map_err(AsyncIoError::Fsync)?;
        if let Some(user_data) = user_data {
            self.completion_list.push_back((user_data, 0));
            self.eventfd.write(1).unwrap();
        }
        Ok(())
    }

    fn next_completed_request(&mut self) -> Option<(u64, i32)> {
        self.completion_list.pop_front()
    }

    fn punch_hole(&mut self, _offset: u64, _length: u64, _user_data: u64) -> AsyncIoResult<()> {
        Err(AsyncIoError::PunchHole(io::Error::other(
            "punch_hole not supported for VMDK",
        )))
    }

    fn write_zeroes(&mut self, _offset: u64, _length: u64, _user_data: u64) -> AsyncIoResult<()> {
        Err(AsyncIoError::WriteZeroes(io::Error::other(
            "write_zeroes not supported for VMDK",
        )))
    }
}

pub fn probe_vmdk(file: &mut File) -> io::Result<bool> {
    let descriptor = match read_descriptor(file) {
        Ok(value) => value,
        Err(err) if err.kind() == io::ErrorKind::InvalidData => return Ok(false),
        Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => return Ok(false),
        Err(err) if err.kind() == io::ErrorKind::Unsupported => return Ok(false),
        Err(err) => return Err(err),
    };

    has_supported_descriptor_header(&descriptor)
}

fn read_descriptor(file: &mut File) -> io::Result<String> {
    let size = file.metadata()?.len();
    if !(4..=VMDK_MAX_DESCRIPTOR_LEN).contains(&size) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "descriptor file size is outside supported range",
        ));
    }

    file.seek(SeekFrom::Start(0))?;
    let mut data = vec![0u8; size as usize];
    file.read_exact(&mut data)?;

    if u32::from_le_bytes(data[0..4].try_into().unwrap()) == VMDK_SPARSE_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "VMDK sparse extents are not supported",
        ));
    }

    std::str::from_utf8(&data)
        .map(ToOwned::to_owned)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("invalid UTF-8: {e}")))
}

fn has_supported_descriptor_header(descriptor: &str) -> io::Result<bool> {
    let mut has_version = false;
    let mut has_type = false;

    for line in descriptor.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        let value = value.trim();

        match key {
            "version" => {
                let version = parse_u32(value)?;
                if !(1..=3).contains(&version) {
                    return Ok(false);
                }
                has_version = true;
            }
            "createType" => {
                let create_type = strip_quotes(value);
                if !matches!(create_type, "monolithicFlat" | "twoGbMaxExtentFlat") {
                    return Ok(false);
                }
                has_type = true;
            }
            _ => {}
        }

        if has_version && has_type {
            return Ok(true);
        }
    }

    Ok(false)
}

fn parse_descriptor(descriptor: &str) -> io::Result<Vec<ParsedExtent>> {
    if !has_supported_descriptor_header(descriptor)? {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "missing required VMDK header fields",
        ));
    }

    let mut extents = Vec::new();
    for line in descriptor.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let Some((first, _)) = line.split_once(char::is_whitespace) else {
            continue;
        };
        if !matches!(first, "RW" | "RDONLY" | "NOACCESS") {
            continue;
        }

        extents.push(parse_extent_line(line)?);
    }

    if extents.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "VMDK descriptor does not contain extents",
        ));
    }
    Ok(extents)
}

fn parse_extent_line(line: &str) -> io::Result<ParsedExtent> {
    let mut parts = line.split_whitespace();
    let access = match parts
        .next()
        .ok_or_else(|| invalid_data("access type missing"))?
    {
        "RW" => VmdkAccess::ReadWrite,
        "RDONLY" => VmdkAccess::ReadOnly,
        "NOACCESS" => VmdkAccess::NoAccess,
        _ => return Err(invalid_data("invalid access type")),
    };

    let sectors: u64 = parts
        .next()
        .ok_or_else(|| invalid_data("sector count missing"))?
        .parse()
        .map_err(|_| invalid_data("invalid sector count"))?;

    if access == VmdkAccess::NoAccess {
        return Ok(ParsedExtent {
            access,
            sectors,
            storage: None,
        });
    }

    let extent_type = parts
        .next()
        .ok_or_else(|| invalid_data("extent type missing"))?;
    if extent_type == "ZERO" {
        return Ok(ParsedExtent {
            access,
            sectors,
            storage: Some(ParsedStorage::Zero),
        });
    }
    if extent_type != "FLAT" {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!("unsupported VMDK extent type: {extent_type}"),
        ));
    }

    let mut quote_split = line.splitn(3, '"').map(str::trim);
    let before_filename = quote_split.next().unwrap();
    let filename = quote_split
        .next()
        .ok_or_else(|| invalid_data("extent filename missing"))?;
    let after_filename = quote_split
        .next()
        .ok_or_else(|| invalid_data("extent filename not terminated"))?;

    if before_filename.split_whitespace().count() != 3 {
        return Err(invalid_data(
            "extent filename appears at unexpected position",
        ));
    }

    let mut after_parts = after_filename.split_whitespace();
    let offset = after_parts
        .next()
        .map_or(Ok(0), |value| value.parse())
        .map_err(|_| invalid_data("invalid extent offset"))?;

    Ok(ParsedExtent {
        access,
        sectors,
        storage: Some(ParsedStorage::Flat {
            filename: filename.to_owned(),
            offset,
        }),
    })
}

fn open_extents(
    descriptor_path: &Path,
    parsed_extents: Vec<ParsedExtent>,
    readonly: bool,
    direct: bool,
) -> io::Result<Vec<VmdkExtent>> {
    let mut extents = Vec::with_capacity(parsed_extents.len());
    let mut disk_offset = 0u64;
    let parent = descriptor_path.parent().unwrap_or_else(|| Path::new("."));

    for parsed in parsed_extents {
        let size = parsed
            .sectors
            .checked_mul(VMDK_SECTOR_SIZE)
            .ok_or_else(|| invalid_data("extent size overflow"))?;
        let next = disk_offset
            .checked_add(size)
            .ok_or_else(|| invalid_data("disk size overflow"))?;
        let storage = match parsed.storage {
            Some(ParsedStorage::Flat { filename, offset }) => {
                let mut opts = OpenOptions::new();
                opts.read(true);
                let can_write = parsed.access == VmdkAccess::ReadWrite && !readonly;
                opts.write(can_write);
                if direct {
                    opts.custom_flags(libc::O_DIRECT);
                }

                let extent_path = absolute_or_join(parent, &filename);
                let file = opts.open(&extent_path)?;
                Some(ExtentStorage::Flat {
                    file,
                    offset: offset
                        .checked_mul(VMDK_SECTOR_SIZE)
                        .ok_or_else(|| invalid_data("extent offset overflow"))?,
                })
            }
            Some(ParsedStorage::Zero) => Some(ExtentStorage::Zero),
            None => None,
        };

        extents.push(VmdkExtent {
            access: parsed.access,
            disk_range: disk_offset..next,
            storage,
        });
        disk_offset = next;
    }

    Ok(extents)
}

fn absolute_or_join(base: &Path, name: &str) -> PathBuf {
    let path = Path::new(name);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

fn strip_quotes(value: &str) -> &str {
    value
        .strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .unwrap_or(value)
}

fn parse_u32(value: &str) -> io::Result<u32> {
    strip_quotes(value)
        .parse()
        .map_err(|_| invalid_data("invalid numeric value"))
}

fn invalid_data(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

fn classify_open_error(err: io::Error) -> BlockError {
    let kind = match err.kind() {
        io::ErrorKind::InvalidData => BlockErrorKind::InvalidFormat,
        io::ErrorKind::Unsupported => BlockErrorKind::UnsupportedFeature,
        _ => BlockErrorKind::Io,
    };
    BlockError::new(kind, err)
}

#[cfg(test)]
mod unit_tests {
    use std::io::Write;

    use vmm_sys_util::tempdir::TempDir;

    use super::*;
    use crate::disk_file::AsyncDiskFile;
    use crate::disk_file::{DiskSize, PhysicalSize};

    #[test]
    fn detect_vmdk_descriptor() {
        let dir = TempDir::new_with_prefix("/tmp/ch").unwrap();
        let path = dir.as_path().join("test.vmdk");
        let mut file = File::create(&path).unwrap();
        writeln!(
            file,
            r#"version=1
CID=fffffffe
parentCID=ffffffff
createType="monolithicFlat"
RW 8 FLAT "data-flat.vmdk" 0"#
        )
        .unwrap();
        drop(file);

        let mut file = File::open(path).unwrap();
        assert!(probe_vmdk(&mut file).unwrap());
    }

    #[test]
    fn reject_sparse_vmdk_header() {
        let dir = TempDir::new_with_prefix("/tmp/ch").unwrap();
        let path = dir.as_path().join("sparse.vmdk");
        let mut file = File::create(&path).unwrap();
        file.write_all(&VMDK_SPARSE_MAGIC.to_le_bytes()).unwrap();
        file.set_len(4096).unwrap();
        drop(file);

        let mut file = File::open(path).unwrap();
        assert!(!probe_vmdk(&mut file).unwrap());
    }

    #[test]
    fn read_write_across_two_flat_extents() {
        let dir = TempDir::new_with_prefix("/tmp/ch").unwrap();
        let descriptor_path = dir.as_path().join("disk.vmdk");
        let extent0_path = dir.as_path().join("extent0-flat.vmdk");
        let extent1_path = dir.as_path().join("extent1-flat.vmdk");

        {
            let mut desc = File::create(&descriptor_path).unwrap();
            writeln!(
                desc,
                r#"# Disk DescriptorFile
version=1
CID=fffffffe
parentCID=ffffffff
createType="twoGbMaxExtentFlat"
RW 8 FLAT "extent0-flat.vmdk" 0
RW 8 FLAT "extent1-flat.vmdk" 0"#
            )
            .unwrap();
        }

        File::create(&extent0_path)
            .unwrap()
            .set_len(8 * VMDK_SECTOR_SIZE)
            .unwrap();
        File::create(&extent1_path)
            .unwrap()
            .set_len(8 * VMDK_SECTOR_SIZE)
            .unwrap();

        let descriptor = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&descriptor_path)
            .unwrap();
        let disk = VmdkDiskSync::new(descriptor, &descriptor_path, false, false).unwrap();
        assert_eq!(disk.logical_size().unwrap(), 16 * VMDK_SECTOR_SIZE);
        assert!(disk.physical_size().unwrap() >= disk.logical_size().unwrap());

        let mut io = disk.create_async_io(128).unwrap();
        let mut payload = vec![0x5a; 1024];
        let split = 256usize;
        let (left, right) = payload.split_at_mut(split);
        let write_iovecs = [
            libc::iovec {
                iov_base: left.as_mut_ptr().cast(),
                iov_len: left.len(),
            },
            libc::iovec {
                iov_base: right.as_ptr() as *mut libc::c_void,
                iov_len: right.len(),
            },
        ];
        let start = (8 * VMDK_SECTOR_SIZE) - 128;
        io.write_vectored(start as libc::off_t, &write_iovecs, 1)
            .unwrap();
        assert_eq!(io.next_completed_request().unwrap(), (1, 1024));

        let mut read_buf = vec![0u8; 1024];
        let (read_left, read_right) = read_buf.split_at_mut(split);
        let read_iovecs = [
            libc::iovec {
                iov_base: read_left.as_mut_ptr().cast(),
                iov_len: read_left.len(),
            },
            libc::iovec {
                iov_base: read_right.as_mut_ptr().cast(),
                iov_len: read_right.len(),
            },
        ];
        io.read_vectored(start as libc::off_t, &read_iovecs, 2)
            .unwrap();
        assert_eq!(io.next_completed_request().unwrap(), (2, 1024));
        assert_eq!(payload, read_buf);
    }
}
