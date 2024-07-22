#![allow(unused_must_use)]

use crate::loader;
use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::svddc::{JobSvd, SVDDC, SVDDCInplace};
use std::path::{Path, PathBuf};
use std::fs::File;
use memmap2::MmapOptions;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use rayon::prelude::*;

struct IncrementalPCA {
    filepath: PathBuf,
    result_path: PathBuf,
    n_spectra: usize,
    n_components: usize,
    fit_batches: Vec<usize>,
    transform_batches: Vec<usize>,
    mean: Array1<f32>,
    singular_values: Array1<f32>,
    explained_variance: Array1<f32>,
    explained_variance_ratio: Array1<f32>,
    n_spectra_seen: f32, 
    variance: Array1<f32>,
    components_: Array2<f32>,
}
   
/// IPCA function.
/// 
/// Input:
/// - path: &PathBuf = filepath to binary containing f32 intensity values
/// - result_path: &PathBuf = output filepath of transformed data
/// - metadata: &loader::Metadata = Metadata from json file
///
/// Output: 
/// - Loading scores used to transform
pub fn incremental_pca(path: &PathBuf, result_path: &PathBuf, metadata: &loader::Metadata) -> Array2<f32> {
    let fit_batches: Vec<usize> = fit_batch_est(metadata.number_of_spectra, metadata.number_of_peaks); 
    let transform_batches: Vec<usize> =  transform_batch_est(metadata.number_of_spectra, metadata.number_of_peaks);
    let mut incremental_pca: IncrementalPCA = IncrementalPCA {
        filepath: path.clone(),
        result_path: result_path.clone(),
        n_spectra: metadata.number_of_spectra,
        n_components: metadata.number_of_peaks,
        mean: Array1::<f32>::zeros(metadata.number_of_peaks),
        singular_values: Array1::<f32>::zeros(metadata.number_of_peaks),
        explained_variance: Array1::<f32>::zeros(metadata.number_of_peaks),
        explained_variance_ratio: Array1::<f32>::zeros(metadata.number_of_peaks),
        n_spectra_seen: 0.0, 
        variance: Array1::<f32>::zeros(metadata.number_of_peaks),
        components_: Array2::<f32>::zeros((0,0)),
        fit_batches,
        transform_batches,
    };

    // fit() iterates over the batches and calls the partial_fit() function for loaded each batch 
    incremental_pca.fit();
    //Transforms the data using self.components_ in batches and writes it to an output file 
    incremental_pca.transform();
    println!("Explained variance ratio:\n{:?}\n", incremental_pca.explained_variance_ratio);
    // Loadings scores are given back.
    incremental_pca.components_.to_owned()
    
}

impl IncrementalPCA {
    ///Iterates over the batches in which per iteration the data of a single batch is loaded converted it to a Array2<f32> and calls the partial_fit function on this data.  
    ///Within each loop iteration a memory map is generated at the offset and length of the current batch. This is done to remove the previously loaded batch from (shared) memory.
    fn fit(&mut self) -> Result<(), std::io::Error> {
        for  batch_index in self.fit_batches.clone().windows(2){
            let mmap = loader::mmap_in_parts(&self.filepath, batch_index[0] as u64, batch_index[1]).unwrap();
            let loaded_bytes: &[u8] = &mmap[..];
            let batch_vec: Vec<f32> = loaded_bytes.par_chunks_exact(4)
                                    .map(|mzs| f32::from_le_bytes(mzs.try_into().unwrap())).collect();
            let batch_array = unsafe{Array2::from_shape_vec_unchecked(((batch_vec.len() as f32 / self.n_components as f32) as usize,
                 self.n_components), batch_vec)}; 
            self.partial_fit(batch_array);  
            
        }
               Ok(())
    }
    
    fn partial_fit(&mut self, x: Array2<f32>) {
        let (mut xt, t, col_mean, col_var, n_total_spectra, spectra_in_batch) = self.incremental_mean_var(x);
        if self.n_spectra_seen > 0.0 {
            let mean_correction: Array2::<f32> = ((self.mean.clone() - &t) *
             ((&self.n_spectra_seen / &n_total_spectra) * (spectra_in_batch)) 
            .sqrt()).insert_axis(Axis(0));
            let s_vt = &self.components_.clone() * self.singular_values.clone().insert_axis(Axis(1)); 
            xt = ndarray::concatenate(Axis(0), &[s_vt.view(), xt.view(), mean_correction.view()]).unwrap();
        }
        let (_, s, vt) = xt.svddc_inplace(JobSvd::Some).unwrap();
        let vt = svd_flip(vt.unwrap());
        if self.n_spectra == (n_total_spectra as usize){
            self.explained_variance = s.mapv(|s: f32| s.powi(2) / (n_total_spectra - 1.0));
            self.explained_variance_ratio = s.mapv(|s: f32| s.powi(2)) / (n_total_spectra * &col_var).sum() ;
        } 
        self.n_spectra_seen = n_total_spectra;
        self.components_ = vt;
        self.singular_values = s;
        self.mean = col_mean;
        self.variance = col_var;
    }


    ///Does the dot product of the original data (in a MMAP) with the loading scores (RAM) parrallellized and writes the results parallelized and async
    ///Tested and exact same results as sequential writing.
    fn transform(&mut self) -> Result<(), std::io::Error>{
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(&self.result_path)
            .expect("Failed to create file");
        let len = (self.n_spectra*self.n_components*4) as u64;
        file.set_len(len);
    
        self.transform_batches.par_windows(2).for_each(|chunk| {
            let mmap = loader::mmap_in_parts(&self.filepath, chunk[0] as u64, chunk[1] as usize).unwrap();
            let loaded_bytes = &mmap[..];
            let batch_vec: Vec<f32> = loaded_bytes.par_chunks_exact(4)
                                    .map(|mzs| f32::from_le_bytes(mzs.try_into().unwrap())).collect();
            let batch_array = Array2::from_shape_vec(((batch_vec.len() as f32 / self.n_components as f32) as usize, self.n_components),batch_vec).unwrap();  
            let xt: Array2::<f32> = batch_array - &self.mean;
            let a = xt.dot(&self.components_.t());
            let bytes_slice: &[u8] = unsafe {
                std::slice::from_raw_parts(a.as_ptr() as *const u8, a.len() * std::mem::size_of::<f32>())
            };
            loader::mmap_write(&file, chunk[0] as u64, chunk[1], bytes_slice);
        });
            
    Ok(())
    }





    /// Incrementally calculates mean, variance, total number of spectra. Also substracts the mean from the data input and gives the number of spectra within the batch.
    fn incremental_mean_var(&self, x: Array2<f32>) -> (Array2<f32>, Array1<f32>, Array1<f32>, Array1<f32>, f32, f32){ 
        // let last_spectra_count: Array1<f32> = Array1::from_elem(x.shape()[1], self.n_spectra_seen) ;
        let last_sum = &self.mean * self.n_spectra_seen;
        let new_sum = x.sum_axis(Axis(0));
        let new_spectra_count = x.shape()[0] as f32;
        let updated_spectra_count = self.n_spectra_seen + new_spectra_count;
        let updated_mean = (&last_sum + &new_sum) / updated_spectra_count;
        let t = &new_sum / new_spectra_count;
        let xt = x-&t;
        let temp = xt.mapv(|i: f32| i.powi(2));
        let correction = xt.sum_axis(Axis(0));  // x̄ 
        let new_unnormalized_variance = temp.sum_axis(Axis(0)) - correction.mapv(|corr: f32| corr.powi(2)) / (new_spectra_count); 
        let last_unnormalized_variance = self.variance.clone() * self.n_spectra_seen;
        let last_over_new_count = self.n_spectra_seen / new_spectra_count;
        let updated_unnormalized_variance = if self.n_spectra_seen == 0.0 {new_unnormalized_variance}
            else {last_unnormalized_variance 
                + new_unnormalized_variance 
                + &last_over_new_count 
                / &updated_spectra_count
                * (last_sum / last_over_new_count - new_sum).mapv(|i: f32| i.powi(2))};
        let updated_variance = updated_unnormalized_variance / updated_spectra_count;
        return (xt, t, updated_mean, updated_variance, updated_spectra_count, new_spectra_count) 
    }
}


fn svd_flip(v: Array2<f32>) -> Array2<f32> {
    let max_values = v.clone().map_axis(Axis(1), |i| max_absolute_value(i.to_owned()));
    let sign = max_values.mapv_into(|i: f32| sign(i));
    let return_v: Array2::<f32> = v * sign.insert_axis(Axis(1));
    return_v
}


pub fn regular_pca<P: AsRef<Path>>(path: P, metadata: &loader::Metadata) -> (Array2<f32>, Array2<f32>) {
    let mmap =  unsafe {
        MmapOptions::new()
                    .map(&File::open(path).unwrap())
                    }.unwrap();


    let loaded_bytes = mmap.get(..).unwrap(); 
    let batch_vec: Vec<f32> = loaded_bytes.par_chunks_exact(4).map(|mzs| f32::from_le_bytes(mzs.try_into().unwrap())).collect();
    let data = Array2::from_shape_vec((metadata.number_of_spectra as usize, metadata.number_of_peaks), batch_vec).unwrap();
    drop(mmap);
    let _col_var = column_variance(&data);
    
    let n = data.shape()[0];
    // Subtract the mean from each peak
    let mean = data.mean_axis(Axis(0)).unwrap();
    let centered_data = data - &mean;
    // Perform singular value decomposition
    let (_u, s, vt) = centered_data.svddc(JobSvd::Some).unwrap();
    let vt = svd_flip(vt.unwrap());
    let eigen_vectors = vt;
    let explained_variance = s.mapv(|s: f32| s.powi(2)) / (n as f32 - 1.0);
    let _explained_variance_ratio = &explained_variance / explained_variance.sum();
    // Project the data onto the principal components
    let projected_data = centered_data.dot(&eigen_vectors.t());

    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .open("results.ripca")
        .expect("Failed to create file");
    let mut writer = BufWriter::new(file);
        for float in projected_data.iter(){
            writer.write_all(&float.to_le_bytes())
                .expect("Failed to write to file");
        }
            // Flush the buffer to ensure all data is written to the file
            writer.flush().expect("Failed to flush buffer");

return (projected_data, eigen_vectors)
}

fn max_absolute_value(array: Array1<f32>) -> f32 {
    let max_index = array
        .iter()
        .enumerate()
        .max_by(|(_, &a), (_, &b)| a.abs().partial_cmp(&b.abs()).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    array[max_index]
}

fn sign(input: f32) -> f32{
    if input > 0.0 {1.0} else if input < 0.0 {-1.0} else{0.0}
}

/// Attempts to find optimal batch sizes. Where each batch is >20 * number of peaks. 
/// This also must include the last batch. The batch size is not allowed to exceed 150_000.
/// For datasets with peaks < 1000 there is a starting offset with the initial estimation.
/// The optimal batch size is hardware and MKL dependant. (Cache size and number of cores used)
/// 
/// Output is a Vec<usize> of the indices of the bytes that need to be loaded.
/// .window(2) is used over the vector containing with the start and end of what needs to be loaded
fn fit_batch_est(spectra: usize, peaks: usize) -> Vec<usize>{
    let minimum_ratio:  usize = 20;
    let peaks_offset = if peaks < 1000{20000 - 20 * peaks} else{0}; 
    let mut batch_size: usize = peaks * 20 + peaks_offset;  // Initial Estimate
    let mut trailing_batch = spectra % batch_size;
    let mut trailing_batch_ratio = trailing_batch / peaks;
    //Attempts to find a batch size with a batch size ratio to peak of >20 that at also has a trailing batch size to peak ratio of >20
    while trailing_batch_ratio < minimum_ratio && trailing_batch != 0{
        let number_of_loops = spectra / batch_size;
        let loop_upperbound = (spectra as f32 / number_of_loops as f32).ceil() as usize;
        let loop_lowerbound = (spectra as f32 / (number_of_loops + 1) as f32).ceil() as usize;
        let trailing_treshold_difference = ((minimum_ratio * peaks - trailing_batch) as f32 / number_of_loops as f32).ceil() as usize;
        if trailing_treshold_difference > 0 && (batch_size - trailing_treshold_difference) >= loop_lowerbound{
            batch_size -= trailing_treshold_difference; 
        } else{
            batch_size = loop_upperbound;
        }
        if batch_size > 150000{
            batch_size = peaks * 20 + peaks_offset;  
            break
        }
        if batch_size >= spectra{
            batch_size = spectra;
            break
        }
        trailing_batch = spectra % batch_size;
        trailing_batch_ratio = trailing_batch / peaks;
    }
    trailing_batch = spectra % batch_size;
    let n_batches = (spectra as f32 /batch_size as f32).floor() as usize;
    let exact = if trailing_batch == 0 {true}else{false};


    let mut batches = vec![batch_size * peaks * 4; n_batches];
    if !exact{
        batches.push(trailing_batch * peaks * 4);
    }
    let batches: Vec<usize> = std::iter::once(0)
    .chain(batches.iter().scan(0, |counter, &i| {
        *counter += i;
        Some(*counter)
    }))
    .collect();
    

    batches
    
}

/// Keeps the total ammount that is loaded stable
/// Loads ~80MB of data at a time. 
/// Has much less effect than the fit_batch_est()
/// Possibly better ways exist for this. 
/// Depends on hardware what the optimal size is.
///
/// Output is a Vec<usize> of the indices of the bytes that need to be loaded.
/// .window(2) is used over the vector containing with the start and end of what needs to be loaded
fn transform_batch_est(spectra: usize, peaks: usize) -> Vec<usize>{
    let batch_size = (20_000_000.0 / peaks as f32).floor() as usize;
    let n_batches = (spectra as f32 / batch_size as f32).floor() as usize; 
    let trailing_batch = spectra % batch_size;
    let exact = trailing_batch == 0;

    let mut batches: Vec<usize> = vec![batch_size * peaks * 4; n_batches];
if !exact {
    batches.push(trailing_batch * peaks * 4);
}

let batches: Vec<usize> = std::iter::once(0)
    .chain(batches.iter().scan(0, |counter, &i| {
        *counter += i;
        Some(*counter)
    }))
    .collect();

    batches
}

fn column_variance(input: &Array2<f32>) -> Array1<f32> {
    let num_columns = input.shape()[1];
    let mut variances = Array1::zeros(num_columns);

    for i in 0..num_columns {
        let column = input.column(i);
        let mean = column.mean().unwrap(); // Calculate the mean of the column
        let variance = column.fold(0.0, |acc, &x| acc + (x - mean).powi(2)); // Calculate sum of squared differences
        variances[i] = variance / (column.len() as f32); // Normalize by column length
    }

    variances
}

