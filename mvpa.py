import numpy as np
import nibabel as nib
import sys
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier as GPC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool
import torch

#Implement a multivoxel Pattern analysis
nR= 13
diam = 6
radius = diam/2

Runs = np.load("Runs.npy")
T1 = nib.load("rnewT1_2.nii").get_fdata()
anatMask = T1 > 60

# If we already have the searchlights computed
if sys.argv[-1] == 'preload':
    searchlights = np.load("SLs.npy")
    print(searchlights.shape)
    nVoxels = searchlights.shape[-1]
else:
    # Get searchlights within sphere
    searchlights = []
    Runs[:,T1 < 60] = 0

    # For calculating voxel distances. Use CUDA to speed this up immensely
    pos = torch.from_numpy(np.array([[l,m,n] for l in range(Runs.shape[1]) for m in range(Runs.shape[2]) for n in range(Runs.shape[3])])).float().cuda()

    maxShape = [0,0,0]
    for i in range(Runs.shape[1]):
        for j in range(Runs.shape[2]):
            for k in range(Runs.shape[3]):
                cent = torch.from_numpy(np.array([i+radius, j+radius, k+radius])).float().cuda()
                searchlights += [Runs[:,(torch.sqrt(((pos - cent)**2).sum(1)) < radius).reshape(Runs.shape[1], Runs.shape[2], Runs.shape[3]).cpu().numpy(),:,:]]
                # Keep tab of greatest number of voxels for zeropadding
                # searchlight centers that don't contain the same number of voxels to make a 
                # nice np.array
                if maxShape[0] < searchlights[-1].shape[1]:
                    maxShape[0] = searchlights[-1].shape[1]
     print(i, maxShape)
    
    # Zero padding searchlight
    for i in range(len(searchlights)):
        temp = np.zeros((Runs.shape[0], maxShape[0], Runs.shape[-2], Runs.shape[1]))
        x = searchlights[i].shape[1]
        temp[:,:x] = searchlights[i]
        searchlights[i] = temp

    # In shape nR x classes, searchlights, voxels of searchlight
    searchlights = np.array(searchlights).squeeze().transpose(1,3,0,2)
    nVoxels = searchlights.shape[-1]
    np.save("SLs.npy",searchlights)
# leave one run out
# trials in shape Run, Run -1, classes, searchlights, voxels of searchlight
# classes in shape Run, Run -1, classes, searchlights

# Make this variables global for multiprocessing
global trials
global classes
global testtrials
trials = np.array([[searchlights[j] for j in range(nR) if j != i] for i in range(nR)])
testtrials = np.array([[searchlights[j] for j in range(nR) if j == i] for i in range(nR)])
classes = np.zeros_like(trials[...,0])
# Conveniently I have my sessions ordered by the last dimension
classes[:,:,1] = 1
testclasses = np.zeros_like(testtrials[...,0])
testclasses[:,:,1] = 1

# Reshape to Run, Examples, Searchlights, voxels of search light
trials = trials.reshape(nR,-1, trials.shape[-2], nVoxels)
testtrials = testtrials.reshape(nR,-1, testtrials.shape[-2], nVoxels)
classes = classes.reshape(nR, -1, classes.shape[-1])
testclasses = testclasses.reshape(nR,-1, testclasses.shape[-1])

def parproc(dat):
    runleftout = []
    size = classes.shape[-1]
    perinc = size//100
    for j in range(classes.shape[-1]):
        if j % perinc == 0:
            print("{0:.3f}".format(j/size), end='\r')
        # If my searchlight is a bunch of zeros label it incorrectly on purpose
        # and skip it
        if trials[dat,:,j].sum() == 0:
            runleftout += [np.array([1,0])]
        else:
            # We can choose any classifier here
            lda = LinearSVC()
            # I have way many more features (voxels) than examples so feature reduction
            # is really important to avoid overparameterization
            pca = PCA(0.9)
            # Standarizing the feature space is important for optimization that
            # way weights would be apriori same magnitude (especially important for
            # gradient optimizations like in SVC; not so much for analytic
            # solution methods like LDA where it's just a projection transform
            # and covariance search).
            scaler = StandardScaler()
            lda.fit(pca.fit_transform(scaler.fit_transform(trials[dat,:,j])), classes[dat,:,j])
            runleftout += [lda.predict(pca.transform(scaler.transform(testtrials[dat,:,j])))]
    return np.array(runleftout).T

# Arguments for multiprocessing, i.e. run selection for validation
datargs = [i for i in range(len(trials))]

# I have a lot of cores on my server so each one can handle each fold of the
# leave one run out analysis
with Pool(nR) as pool:
    runleftout = pool.map(parproc, datargs)

# Get Accuracy
runleftout = np.array(runleftout).reshape(nR, 2, Runs.shape[1], Runs.shape[2], Runs.shape[3])
testclasses = testclasses.reshape(nR,2,Runs.shape[1], Runs.shape[2], Runs.shape[3])
accuracy = (runleftout == testclasses).mean(0)

# Marginal accuracies of first and second class
RLO0 = nib.Nifti1Image(accuracy[0], affine=np.load("affine.npy"))
RLO1 = nib.Nifti1Image(accuracy[1], affine=np.load("affine.npy"))
# Joint accuracies
RLOAll = nib.Nifti1Image(accuracy.mean(0), affine=np.load("affine.npy"))

nib.save(RLO1, "SVCRLO1_new7.nii")
nib.save(RLO0, "SVCRLO0_new7.nii")
nib.save(RLOAll, "SVCRLOAll_new7.nii")
