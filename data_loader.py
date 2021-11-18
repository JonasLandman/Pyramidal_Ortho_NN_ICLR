import os, sys, inspect
currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parent2dir = os.path.dirname(parentdir)
parent3dir = os.path.dirname(parent2dir)
parent4dir = os.path.dirname(parent3dir)
sys.path.append(parent4dir + "/qcware_algorithms")

import requests
import zipfile
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
from skimage.transform import rescale
from sklearn.preprocessing import StandardScaler
import pickle
import gzip



def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


# In[3]:


def load_dataset(dataset_name='chest'):
    gdrive_keys = {
        'breast': '1y7-GfDFqNmEwTE4bXf7JztOvCqMzQ659',
        'chest': '1qVzlzdFTw-mlGahQlQ_IMr7zVkkM9ARi',
        'derma': '1WE-iSzU3XJRe6HSe1NcWfQgKKUrua2ez',
        'oct': '1oeRHogBg1svVatjuW7cDSymUgu9wBDZG',
        'axial':'1L3kPzvyoiMmUmWnzBFnscVDPrs1089Pc',
        'coronal':'18ltnQoDlXMrX91U7vG6AbFvYTvqHxvdZ',
        'sagittal': '1sUuPUr8V3E5wYPZc-0SLPeFWbvUzJ82R',
        'path': '19jo5PuDK8GiHQVWXdry0rkpx3AxFvhYC', 
        'pneumonia':'1h64Sp5VMDAz_nAlPPFsbqRLv01_hWXBH',
        'retina': '1bhVbnkCflwimYglB2rHeW4Y_gfoaZQWG'
    }

    if not os.path.isdir('data/'):
        get_ipython().system('mkdir data')

    file_id = gdrive_keys[dataset_name]

    destination = './data/' + dataset_name + '.zip'

    if os.path.isdir('data/' + dataset_name):
        pass

    else:
        print('Downloading...')

        download_file_from_google_drive(file_id, destination)

        print('Extracting...')
        with zipfile.ZipFile('data/' + dataset_name + '.zip', 'r') as zip_ref:
            zip_ref.extractall('data/' + dataset_name)

        print('Done.')


def load_data(dataset,
              classes=None,
              samples=None,
              dim_scale=None,
              dim_red_method='PCA',
              zipped=True,
              balanced = False):
    """
    Load data from MedMNIST dataset. If optional arguments not specified,
    default data parameters are used.
    
    Args: 
    dataset: (str) 'breast', 'chest', 'derma', 'oct', 'axial', 'coronal', 'sagittal', 'path', 'pneumonia', 'retina'
        string specifiying which dataset to load.
    n_classes: (int or list)
        Number of classes to be included in dataset. If None, set to match dataset.
        If list specified, then loads just these classes. Eg. [0,2] gives just classes 0 and 2. 
    num_samples: (array) (integers or floats)
        Specify number of training, test and validation data samples to use. 
        If entries are integers e.g [100,50,50], return 100,50,50 training, test, validation samples resp.
        If entries are floats eg. [0.5,0.3,0.2], return specified ratios of training, test, validation samples resp.
        If None, return all data.      
    data_dim: (float), (|0,1)). 
        Set the dimension scale factor of each image. 
        For 'average' method, 0.25 reduces a 784 (28 * 28)  image to 49 (7 * 7) .
        For 'PCA' method, 0.25 reduces 784 (28 * 28) image 196 (14*14).
        Note, image and labels are returned as flattened column vectors.  
    dim_red_method: (str) 'average' or 'PCA'
        Set the method with which to perform dimensionality reduction. 
        'PCA' is (Principal Component Analysis).
        'average' is image rescaling (via averaging). 
    zipped: (Bool) 
        Default True. Return data zipped into (X,y) pairs. 
        Otherwise, returns six arrays in the following order: tr_x, tr_y, te_x, te_y, va_x, va_y
    balanced: Gives same number of samples for each class in a dataset
    verbose: print breakdown of classes by number 
    """
    
    #Load MNIST dataset:
    
    if dataset == 'MNIST': 
        f = gzip.open('mnist.pkl.gz', 'rb')
        training_data, test_data, validation_data = pickle.load(f,
                                                                encoding="latin1")
        f.close()
        
        tr_x = training_data[0]
        tr_y = np.reshape(training_data[1], (-1,1))
        
        te_x = test_data[0]
        te_y = np.reshape(test_data[1],(-1,1))
        
        va_x = validation_data[0]
        va_y = np.reshape(validation_data[1], (-1,1))    

    #Load MedMNIST datasets
    elif dataset in [
            "breast", "chest", "derma", "oct", "axial", "coronal", "sagittal",
            "path", "pneumonia", "retina"
    ]:

        #Load the dataset by downloading into seperate folder. (Only if  not already downloaded in right location.)
        load_dataset(dataset)

        #Load training data
        tr_x = np.load('./data/' + dataset + '/train_images.npy')
        tr_y = np.load('./data/' + dataset + '/train_labels.npy')

        #Load test data
        te_x = np.load('./data/' + dataset + '/test_images.npy')
        te_y = np.load('./data/' + dataset + '/test_labels.npy')

        #Load validation data
        va_x = np.load('./data/' + dataset + '/val_images.npy')
        va_y = np.load('./data/' + dataset + '/val_labels.npy')
        
    else:
        raise ValueError("dataset should be one of: 'breast', 'chest', 'derma', 'oct', 'axial', 'coronal', 'sagittal', 'path', 'pneumonia', 'retina'")
    
   # ----------------pick only the relevant classes


#     all_images = np.concatenate((tr_x, va_x, te_x))
#     all_labels = np.concatenate((tr_y, va_y, te_y))
# #     

    allowed_values = np.array(classes, dtype=object)
    if type(classes[0]) is list:
        allowed_values_flat = np.array([j for sub_list in classes for j in sub_list])
    else:                                              
        allowed_values_flat = allowed_values

    indices_tr = np.where(tr_y == allowed_values_flat)[0]
    tr_x = tr_x[indices_tr]
    tr_y = tr_y[indices_tr]
    
    indices_te = np.where(te_y == allowed_values_flat)[0]
    te_x = te_x[indices_te]
    te_y = te_y[indices_te]
    
    indices_va = np.where(va_y == allowed_values_flat)[0]
    va_x = va_x[indices_va]
    va_y = va_y[indices_va]

    all_images = np.concatenate((tr_x, va_x, te_x))
    all_labels = np.concatenate((tr_y, va_y, te_y))

    for i in range(len(all_labels)):
        all_labels[i] = np.where(all_labels[i] == allowed_values_flat)
    
    if type(classes[0]) is list:
        for i in all_labels:
            if i[0] in classes[0]:
                i[0]=0
            else:
                i[0]=1

    tr_y = all_labels[:len(tr_y)]
    te_y = all_labels[-len(te_y):]
    va_y = all_labels[len(tr_y):-len(te_y)]




 
     #----------------Perform PCA on full dataset, if applicable.--------------
   
    
    if (dim_red_method == 'PCA'): # and dim_scale != 784):
        
        if type(dim_scale) == float:
            raise ValueError("dim_scale must be int for PCA")
      
        else: 
            all_images = np.concatenate((tr_x, va_x, te_x))

            all_images = [np.ndarray.flatten(i) for i in all_images]
            all_images = StandardScaler().fit_transform(all_images)
            pca = PCA(n_components = int(dim_scale))
            all_images = pca.fit_transform(all_images)
            all_images = StandardScaler().fit_transform(all_images)
            
            tr_x = all_images[:len(tr_x)]
            te_x = all_images[-len(te_x):]
            va_x = all_images[len(tr_x):-len(te_x)]
            
            all_images = np.concatenate((tr_x, va_x, te_x))

            
    elif (dim_red_method == 'average'):
        
        if np.sqrt(dim_scale)%1 == 0:
            all_images = np.concatenate((tr_x, va_x, te_x))

            all_images = [rescale(np.reshape(i,(28,28)),np.sqrt(dim_scale)/28) for i in all_images]
            all_images = np.array([np.reshape(i,-1) for i in all_images])

            tr_x = all_images[:len(tr_x)]
            te_x = all_images[-len(te_x):]
            va_x = all_images[len(tr_x):-len(te_x)]
            
            all_images = np.concatenate((tr_x, va_x, te_x))
            
        else:
            print("Enter a perfect square as dim_scale")

        
            
    #-----
    
    if balanced==False:
        
        tr_size = samples[0]
        te_size = samples[1]
        va_size = samples[2]

        
        tr_x = tr_x[:tr_size]
        tr_y = tr_y[:tr_size]
        te_x = te_x[:te_size]
        te_y = te_y[:te_size]
        va_x = va_x[:va_size]
        va_y = va_y[:va_size]
        
        #print('tr_x',len(tr_x),tr_x[0:10], 'tr_y', len(tr_y),tr_y[0:10], 'te_x',len(te_x), te_x[0:10],'te_y',len(te_y),te_y[0:10])

    
    else:
        
        locations_tr = [[] for i in range(len(allowed_values_flat))]
        locations_te = [[] for i in range(len(allowed_values_flat))]
        locations_va = [[] for i in range(len(allowed_values_flat))]
        
        for index in range(len(allowed_values_flat)):
            locations_tr[index].append(np.where(tr_y == index)[0])
            locations_te[index].append(np.where(te_y == index)[0])
            locations_va[index].append(np.where(va_y == index)[0])
        
        
        tr_size = samples[0]
        te_size = samples[1]
        va_size = samples[2]
        
        training_x = [[] for i in range(len(allowed_values_flat))]
        test_x = [[] for i in range(len(allowed_values_flat))]
        val_x = [[] for i in range(len(allowed_values_flat))]
        training_y = [[] for i in range(len(allowed_values_flat))]
        test_y = [[] for i in range(len(allowed_values_flat))]
        val_y = [[] for i in range(len(allowed_values_flat))]
        
        for index in range(len(allowed_values_flat)):
            training_x[index] = tr_x[locations_tr[index][0]][:samples[0]]
            test_x[index] = te_x[locations_te[index][0]][:samples[1]]
            val_x[index] = va_x[locations_va[index][0]][:samples[2]]
            training_y[index] = tr_y[locations_tr[index][0]][:samples[0]]
            test_y[index] = te_y[locations_te[index][0]][:samples[1]]
            val_y[index] = va_y[locations_va[index][0]][:samples[2]]
            
#         print(locations_tr[1][0], tr_x[locations_tr[1][0]][:samples[0]])
#         print('y', locations_tr[1][0], tr_y[locations_tr[1][0]][:samples[0]])
        
        tr_x = np.concatenate(training_x)
        te_x = np.concatenate(test_x)
        va_x = np.concatenate(val_x)
        tr_y = np.concatenate(training_y)
        te_y = np.concatenate(test_y)
        va_y = np.concatenate(val_y)
        
    
    
#     print('tr_x',len(tr_x),tr_x[0:10], 'tr_y', len(tr_y),tr_y[0:10], 'te_x',len(te_x),te_x[0:10],'te_y',len(te_y),te_y[0:10])


    tr_x = [np.reshape(np.ndarray.flatten(i), (-1, 1)) for i in tr_x]
    te_x = [np.reshape(np.ndarray.flatten(i), (-1, 1)) for i in te_x]
    va_x = [np.reshape(np.ndarray.flatten(i), (-1, 1)) for i in va_x]

    #Express integer y_label as column vector

    tr_y = [vectorized_result(i[0], len(classes)) for i in tr_y]
    te_y = [vectorized_result(i[0], len(classes)) for i in te_y]
    va_y = [vectorized_result(i[0], len(classes)) for i in va_y]

    #If zipped, express seperate lists as (X,y) pairs
    if zipped == True:
        training_data = list(zip(tr_x, tr_y))
        test_data = list(zip(te_x, te_y))
        validation_data = list(zip(va_x, va_y))
        return training_data, test_data, validation_data
    else:
        return (tr_x, tr_y, te_x, te_y, va_x, va_y)

    
#### Miscellaneous functions
def vectorized_result(j, num_classes):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.
    """
    
    e = np.zeros((num_classes, 1))
    e[j] = 1.0
    return e
