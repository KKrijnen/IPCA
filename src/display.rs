use ndarray::Array2;
use rayon::prelude::*;
use crate::loader;
use image::{Rgba, RgbaImage, ImageBuffer};
use colorgrad;
use std::path::PathBuf;
use std::fs;


fn convert_to_viridis(value: f32) -> Rgba<u8> {
    let viridis = colorgrad::viridis();
    Rgba(viridis.at(value as f64).to_rgba8())
}

fn convert_to_greyscale(value: f32) -> Rgba<u8> {
    let greyscale = colorgrad::CustomGradient::new()
    .html_colors(&["#000000", "#FFFFFF"])
    .build().unwrap();
    Rgba(greyscale.at(value as f64).to_rgba8())
}

fn array_to_image(array: &Array2<f32>, viridis: bool) -> RgbaImage {
    let width = array.ncols();
    let height = array.nrows();

    // Create a new RgbImage with the same dimensions as the array
    let mut image_buffer = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(width as u32, height as u32);

    // Iterate over each pixel in the image buffer and set its color based on the array value
    for (x, y, pixel) in image_buffer.enumerate_pixels_mut() {
        let value = array[[y as usize, x as usize]];
        let color = if viridis == true {convert_to_viridis(value)} else{convert_to_greyscale(value)};    
        *pixel = color;
    }

    image_buffer
}
/// Creates viridis colormap images of each principal component in the result binary.
/// Naive and relatively slow way of writing .png images by loading the data, making the minimum value 0.0 and the maximum 1.0 and generate a 3d rgb matrix.
/// Depending on RAM, Large files (~256.000.000 pixels) may require sequential writing due to RAM limitations. 
/// Issues may start occuring at extremely large files. >>256.000.000 pixels. 
///
/// Input:
/// - metadata: loader::Metadata
/// - path_result_bin: &PathBuf
/// - viridis: bool --> true = viridis, false = greyscale
/// - par: bool --> true = paralellized image writing, false = sequential image writing. Sequential required for large images.
pub fn save_images(metadata: loader::Metadata, path_result_bin: &PathBuf, viridis: bool, par: bool){
    let mut output_folder_path = std::env::current_dir().unwrap();
    output_folder_path.push("image_output\\");
    let _ = fs::create_dir(output_folder_path);
    
    if par == false {
        (0..metadata.number_of_peaks).into_iter().for_each(|i|{
        let mmap = loader::mmap(path_result_bin).unwrap();
        let bytes_to_load = mmap_indexing(metadata.number_of_spectra, metadata.number_of_peaks, i);
        let loaded_bytes = bytes_to_load.iter().map( |i| mmap[*i] ).collect::<Vec<_>>();
        let batch_vec: Vec<f32> = loaded_bytes.chunks_exact(4).map(|eigenvalues| f32::from_le_bytes(eigenvalues.try_into().unwrap())).collect();
        let component_min = find_min_value(&batch_vec);
        let component_max = find_max_value(&batch_vec);
        let batch_vec: Vec<f32> = batch_vec.iter().map(|i| ((i - component_min) / (component_max - component_min))).collect();
        let mut spectrum_counter: usize = 0;
        for region in &metadata.region_names{
            let mut image_array: Array2::<f32> = Array2::zeros((metadata.region[region].x_max+1, metadata.region[region].y_max+1));
            for pixel in &metadata.region[region].pixels {
                image_array[[pixel.x, pixel.y]] = batch_vec[spectrum_counter];
                spectrum_counter += 1;
            }
            let img = array_to_image(&image_array, viridis);
            let output_path = format!("image_output/{}_PC_{}.png", region.clone(), i+1);
            let _ = img.save(output_path);
        }
    })}
    else{
        (0..metadata.number_of_peaks).into_par_iter().for_each(|i|{
            let mmap = loader::mmap(path_result_bin).unwrap();
            let bytes_to_load = mmap_indexing(metadata.number_of_spectra, metadata.number_of_peaks, i);
            let loaded_bytes = bytes_to_load.iter().map( |i| mmap[*i] ).collect::<Vec<_>>();
            let batch_vec: Vec<f32> = loaded_bytes.chunks_exact(4).map(|eigenvalues| f32::from_le_bytes(eigenvalues.try_into().unwrap())).collect();
            let component_min = find_min_value(&batch_vec);
            let component_max = find_max_value(&batch_vec);
            let batch_vec: Vec<f32> = batch_vec.iter().map(|i| ((i - component_min) / (component_max - component_min))).collect();
            let mut spectrum_counter: usize = 0;
            for region in &metadata.region_names{
                let mut image_array: Array2::<f32> = Array2::zeros((metadata.region[region].x_max+1, metadata.region[region].y_max+1));
                for pixel in &metadata.region[region].pixels {
                    image_array[[pixel.x, pixel.y]] = batch_vec[spectrum_counter];
                    spectrum_counter += 1;
                }
                let img = array_to_image(&image_array, viridis);
                let output_path = format!("image_output/{}_PC_{}.png", region.clone(), i+1);
                let _ = img.save(output_path);
            }
        })
    }
}


//indexes into the mmap by component
fn mmap_indexing(n_samples: usize, n_components: usize, component_number: usize) -> Vec<usize> {
    let mut indices: Vec<usize> = Vec::with_capacity(n_samples * 4);

    for i in 0..n_samples {
        let base_index = i * n_components * 4 + component_number * 4;
        indices.extend_from_slice(&[base_index, base_index + 1, base_index + 2, base_index + 3]);
    }
    indices
}

fn find_max_value(values: &[f32]) -> f32 {
    *values.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
}

fn find_min_value(values: &[f32]) -> f32 {
    *values.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
}