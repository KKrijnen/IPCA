use serde::Deserialize;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use memmap2::{Mmap, MmapOptions};
use std::collections::HashMap;
use std::io;


#[derive(Deserialize, Debug)]
pub struct Metadata {
    pub region_names: Vec<String>,
    pub number_of_spectra: usize,
    pub number_of_peaks: usize,
    pub peak_list: Vec<f32>,
    pub region: HashMap<String, RegionStruct>,
    
}

#[derive(Deserialize,Debug)]
pub struct RegionStruct {
    pub number_of_spectra: usize,
    pub x_max: usize,
    pub y_max: usize,
    pub pixels: Vec<PixelStruct>,
}

#[derive(Deserialize,Debug)]
pub struct PixelStruct {
    pub x: usize,
    pub y: usize
}

pub fn read_json<P: AsRef<Path>>(path: P) -> Result<Metadata, Box<dyn Error>> {
    // Open the file in read-only mode with buffer.
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    // Read the JSON contents of the file as an instance of `Metadata`.
    let metadata: Metadata = serde_json::from_reader(reader)?;
    Ok(metadata)
}

pub fn mmap_in_parts<P: AsRef<Path>>(path: P, start: u64, end:usize) -> Result<Mmap, std::io::Error>{
    unsafe {
        MmapOptions::new()
                    .offset(start)
                    .len(end - start as usize)
                    .map(&File::open(path).unwrap())
    }
}

pub fn mmap_write(file: &File, start: u64, end:usize, data: &[u8]){
    let mut mmap = unsafe {
        MmapOptions::new()
                    .offset(start)
                    .len(end - start as usize)
                    .map_mut(file).unwrap()
                    
        };
    
    mmap.copy_from_slice(data);
    let _ = mmap.flush_async();
}

pub fn mmap<P: AsRef<Path>>(path: P) -> Result<Mmap, std::io::Error>{
    unsafe {
        MmapOptions::new()
                    .map(&File::open(path).unwrap())
    }

    
    
}

pub fn file_input() -> (PathBuf, PathBuf, PathBuf){
    println!("Write file name without the extension name:");
    let mut file_name: String = String::new();
    io::stdin()
        .read_line(&mut file_name)
        .expect("Failed to read line");
    let file_name = file_name.trim();

    let json = std::path::PathBuf::from(file_name.to_owned() + ".json");
    let int_bin = std::path::PathBuf::from(file_name.to_owned() + ".bin");
    let result_bin = std::path::PathBuf::from(file_name.to_owned() + "_result.bin");
    (json, int_bin, result_bin)
}

