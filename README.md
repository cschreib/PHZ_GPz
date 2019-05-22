PHZ_GPz Project
===============

This is the euclidised version of the GPz photometric redshift code. The original version, written in Matlab, can be found here:

https://github.com/OxfordML/GPz

and a less up-to-date python version can be found here:

https://github.com/mdevl/GPz

This repository hosts a new version of the code in C++. For questions, please contact Corentin Schreiber (corentin.schreiber@physics.ox.ac.uk).


Outline of the fit procedure as implemented in original GPz
===========================================================

 * **Training**:
   * Read in the training set
   * Compute weights to homogenize coverage in redshift space
   * Split data in training/validation sets
   * Calibrate GP (minimize GP logLikelihood wrt hyperparameters)
     * Iterate with BFGS optimizer, checking if logLikelihood continues to improve on validation set, and if not stop optimization

 * **Prediction**:
   * Read in input photometry + errors
   * Apply GP prediction


Things to figure out
====================

 * How much to make configurable? Aim for configurable most options, or keep it simple and only implement what we know works best?
 * ~~What is the data model?~~ (delegated to Primal framework)
   * ~~What format should we expect input photometry?~~ (delegated to Primal framework)
     * ~~Single catalog, or streamed chunks?~~ (delegated to Primal framework)
     * ~~FITS or ASCII?~~ (delegated to Primal framework)
     * Fluxes of mags?
     * Are we given individual uncertainties + global calibration uncertainties? (not very important at this stage)
     * Anything else?
   * ~~How training is handled in pipeline?~~ (delegated to Primal framework)
     * ~~Will training will be done in separate call to the program, to be reused?~~ (yes and no, program to be wrapped as Phython class, training and predictions are different function calls)
     * Do we need to provide interface to save/load the model without re-training?
     * ~~Do we decide how training data is used? Split sample in RA/Dec, ... etc.~~ (delegated to Primal framework)
   * What format for output?
     * ~~FITS or ASCII?~~ (delegated to Primal framework)
     * ~~Produce full p(z)?~~ (yes)
     * ~~How?~~ (delegated to Primal framework)


Linear algebra tools needed for implementation
==============================================

 * Minimizer: (L?)BFGS
 * Matrix inverse
 * Singular Value Decomposition (SVD)
 * Principal Component Analysis (PCA)
