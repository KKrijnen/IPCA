Modifications made:

I)
Within the provided scikit-learn fork, the following edits have been made in the partial_fit() function within the _incremental_pca.py file: 

... Line 322
        # Update stats - they are 0 if this is the first step
        col_mean, col_var, n_total_samples = _incremental_mean_and_var(
            X,
            last_mean=self.mean_,
            last_variance=self.var_,
            last_sample_count=np.repeat(self.n_samples_seen_, X.shape[1]),
        )
>>      col_mean = col_mean.astype(X.dtype)
>>      col_var = col_var.astype(X.dtype)
>>      n_total_samples = n_total_samples[0].astype(X.dtype)
         # Whitening
        if self.n_samples_seen_ == 0:
            # If it is the first step, simply whiten X
            X -= col_mean
...
The lines annotated with >> are added to the original version.
This edit will ensure that the dtype does not change after the first iteration.
The wheel of this edited fork is provided.  

II) 
The addition of the batch_est_fit() function which attempts to find a batch size with a size to peak ratio of >19 including the final batch.
If after the initial batch_size of 19 * features the size of the last batch that has a peak ratio <19. 
This function will search for larger batch sizes that have a last batch that has a peak ratio >19.



III)
Instead of using transform on the entire data set (which is the default for the .fit_transform() function) increments as well.
Batches with a size of 80MB are chosen (80_000_000 / 4 / features). And then iterated over while each time the transformation results are saved to the disk.

