#![allow(dead_code)]
#![allow(unused_imports)]

mod loader;
mod incremental_pca;
mod display;
use std::time;
use std::env;



fn main(){
  // Envoronment Variables
  env::set_var("KMP_AFFINITY", "granularity=core,scatter,0,0");
  env::set_var("KMP_BLOCKTIME", "10");
  env::set_var("KMP_DYNAMIC_MODE", "load_balance");
  env::set_var("OMP_WAIT_POLICY", "PASSIVE");

  //Asks for file name
  let (path_json, path_int_bin, path_result_bin) = loader::file_input();
  println!("\nLoading Metadata\n");

  // Loads metadata, can take quite long due to the inclusion of pixel coordinates for the saving of component images.  
  let metadata: loader::Metadata = loader::read_json(path_json).unwrap();
  println!("Metadata loading complete, starting IPCA\n");
  let start_time = time::Instant::now();

  //Incremental PCA function. Saves transformed data to .results as a binary containing float-32 numbers
  let _ = incremental_pca::incremental_pca(&path_int_bin,&path_result_bin, &metadata);
  let elapsed_time = start_time.elapsed();
  println!("IPCA run-time: {:?}\n", elapsed_time);
  
  //Saves the component images, changes to sequential writing of images when there are > 5_000_000 pixels
  println!("Converting component values to images\n");
  let par = metadata.number_of_spectra > 10_000_000;
  display::save_images(metadata,&path_result_bin, true, par);
}

