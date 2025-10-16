
   
"""
Gaussian Mixture Surface Explanation:

This section of the code constructs a Gaussian distributions 
centered on each input vector. Conceptually, you can think of the 
model as a landscape made of multiple Gaussian “hills” or peaks:

1. Each input vector is the mean of a Gaussian, forming the center of a hill.
2. The covariance of the Gaussian determines the spread of the hill. 
Here, all Gaussians are spherical with identical covariance.
3. Each Gaussian is assigned a weight (scale) based on its isolation:
- Gaussians farther from other means get higher weights.
- Raising these scores to a power amplifies differences between isolated 
    and crowded Gaussians.
4. The mixture surface is computed by summing all Gaussians together:
- Overlapping hills add to form taller peaks.
- The result is a smooth density landscape over the input space.

Candidate vectors are then evaluated on this landscape:
- The Gaussian mixture PDF at a candidate vector gives its “height” 
on the surface.
- Higher values correspond to vectors lying in dense regions 
or near high-weight Gaussians.
- Lower values correspond to vectors in valleys, far from the peaks.

Finally, scores are converted into percentiles:
- This normalizes the density values, indicating relative ranking 
of candidates on the mixture surface.
- Percentiles can be interpreted as the likelihood of a vector 
relative to all sampled candidates.

In short: the model acts as a probabilistic 
surface, and each candidate vector’s score tells us how “high” 
it sits on that surface.
"""

import numpy as np
from scipy.stats import multivariate_normal, rankdata
from scipy.spatial.distance import mahalanobis

def compute_pdf(mean, cov, allVectors, scale):
    gaussian_pdf = multivariate_normal(mean=mean, cov=cov)
    return scale * gaussian_pdf.pdf(allVectors)


def runCalc(inputVectors, covParam, validation, span):
    inputVectors = np.array(inputVectors)
    N, d = inputVectors.shape

    # -------------------------------------------------------
    # 1. Build Gaussian parameters (means and covariances)
    # -------------------------------------------------------
    gaussian_params = []
    #building Gaussian parameters
    #covariance is identify matrix scaled by covParam (a determined hyperparameter)
    #this makes each Gaussian spherical with the same spread in all directions.
    cov = np.eye(d) * covParam  # same covariance for all Gaussians
    inv_cov = np.linalg.inv(cov)  #inverse covariance matrix needed for computing Mahalanobis distance
    gaussian_params = [(inputVectors[i], cov) for i in range(N)] #store the mean and covariance of each Gaussian

    # print(gaussian_params)
    # -------------------------------------------------------
    # 2. Compute Mahalanobis-based isolation weights
    # -------------------------------------------------------
    mScoreList = []
    #for each Gaussian, calculate the Mahalanobis distance to every other Gaussian, which gives a measure of how isolated / separated each Gaussian is relative to the others.
    for z in range(len(gaussian_params)):
        mean1 = gaussian_params[z][0]
        iScore = 0 #initializing isolation score for this Gaussian
        for q in range(len(gaussian_params)):
            if q != z: #skip comparing Gaussian to itself
                mean2 = gaussian_params[q][0]
                mahalanobis_score = mahalanobis(mean1, mean2, inv_cov) #Compute the Mahalanobis distance between mean1 and mean2.
                iScore += mahalanobis_score #accumulate the distance into the isolation score.

        mScoreList.append(float(iScore))

    mScoreList2 = np.array(mScoreList) ** 3.5 #exaggerate differences between scores by raising them to the power of 3.5.
    #the more isolated each vector is, the higher its weight will be
    mScoreList = mScoreList2
    #storing information of each Gaussian as a tuple of (mean, covariance, isolation score) 
    gpNew = [(gaussian_params[i][0], gaussian_params[i][1], mScoreList[i]) for i in range(len(gaussian_params))]
    gaussian_params = gpNew

    # -------------------------------------------------------
    # 3. Generate candidate vectors
    # -------------------------------------------------------
    # Total number of random vectors to sample = 7000 per input vector.
    #num_samples = 7000 * len(inputVectors)
    # Compute the bounds of uniform sampling windows for each input vector
    mins = inputVectors - span * covParam 
    maxs = inputVectors + span * covParam
    # Sample candidate vectors uniformly within each hypercube
    # For each input vector (i from 0 to N-1):
    # - Draw 7000 random vectors in d dimensions, uniformly distributed within [mins[i], maxs[i]].
    # - Stack all samples from all input vectors vertically into one large array of shape (7000*N, d).
    all_samples = np.vstack([
        np.random.uniform(low=mins[i], high=maxs[i], size=(7000, d)) for i in range(N)
    ])
    # Add validation vectors (if provided) to the candidate pool
    if validation is not None and len(validation) > 0:
        all_vectors = np.vstack([all_samples, validation])
    else:
        all_vectors = all_samples

    #Initialize an array to hold scores for each sampled vector.
    random_scores = np.zeros(len(all_vectors))

    # -------------------------------------------------------
    # 4. Compute Gaussian mixture PDF values
    # -------------------------------------------------------
    for v in range(len(inputVectors)):
        if len(inputVectors) == 1:
            current_result = compute_pdf(gpNew[v][0], gpNew[v][1], all_vectors, 1)
        else:
            current_result = compute_pdf(gpNew[v][0], gpNew[v][1], all_vectors, gpNew[v][2])
        random_scores += current_result

    ranks = rankdata(random_scores, method='average')  # or 'ordinal', 'dense', etc.

    # Convert ranks to percentiles
    percentiles = (ranks - 1) / (len(random_scores) - 1)

    return all_vectors, percentiles, gaussian_params