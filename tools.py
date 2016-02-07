
# coding: utf-8

# ## Modules

# In[1]:

from modules import *


# ##Methods

# In[25]:

matplotlib.rc('font', **{'family' : 'sans-serif',
        'serif'  : 'Helvetica Neue',
        'weight' : 'normal',
        'size'   : 16})


# In[ ]:

def similar(a, b):
    '''
    Computes simlarity of two strings a and b using Python SequenceMatcher'''
    return SequenceMatcher(None, a, b).ratio()


# In[11]:

def bqutil_set_project(project_id):
    if project_id not in ['mitx-research', 'mitx-data', 'harvardx-data']:
        return
    f = open("/Users/cgn/Dropbox (MIT)/cgn/bq/edx2bigquery/edx2bigquery/edx2bigquery_config.py", 'w')
    text = '''PROJECT_ID = "{project_id}"
auth_key_file = "USE_GCLOUD_AUTH"
auth_service_acct = None'''.format(project_id=project_id)
    f.write(text)
    f.close


# In[23]:

def bqutil_get_table(dataset_id, list_of_table_ids, path, gs_folder=None, project_id='mitx-research'):
    '''
    Downloads large tables using Google Storage.
    dataset_id: string of dataset_id, also supports a list of dataset_id strings 
    gs_folder: Specify a folder if persistent storage in Google Storage is needed. 
    
    Need to add support for extracting and combining sharded tables
    '''
    
    persistent = True
    
    if gs_folder is None:
        gs_folder = 'temp'
        persistent = False
    
    if type(dataset_id) == str:
        list_of_dataset_ids = [dataset_id for i in range(len(list_of_table_ids))]
    else:
        list_of_dataset_ids = dataset_id
        
    for i in range(len(list_of_table_ids)):
        table_id = list_of_table_ids[i]
        dataset_id = list_of_dataset_ids[i]
        gsfn = 'gs://mitx-research/' + gs_folder + '/' + dataset_id + "_" + table_id + '*.csv.gz'
        bqutil.extract_table_to_gs(dataset_id=dataset_id, table_id=table_id, gsfn=gsfn, format='csv', do_gzip=True, project_id=project_id)
   
    # Download all tables to path
    gsfn = 'gs://mitx-research/' + gs_folder
    cmd = "gsutil -m cp -r " + gsfn + " " + path
    os.system(cmd)
    
    if not persistent:
        #If user did not ask for persitent storage in gs, remove files   
        cmd = 'gsutil -m rm -r ' + gsfn
        os.system(cmd)
    
    


# In[1]:

def bqutil_df2bq(df, schema, project_id, dataset_id, table_id):
    '''Uploads a dataframe to a bigquery table
    Example schema:
        schema = [{'type': 'STRING', 'name': 'cameo_candidate'},
              {'type': 'STRING', 'name': 'shadow_candidate'},
              {'type': 'FLOAT', 'name': 'similarity'},]
    Example usage:
        df2bq(df, schema, 'mitx-research', '0_cgn_sample', table)
    '''
    
    csv_filename = table_id + '.csv' 
    df.to_csv(csv_filename, header = True, index = False)
    schema_filename = table_id + "__schema.json"
    open(schema_filename, 'w').write(json.dumps(schema, indent=4))
    
    address = project_id + ':' + dataset_id + '.' + table_id
    #Remove table first, otherwise it will append
    try:
        bqutil.delete_bq_table(dataset_id, table_id, project_id)
        print 'Overwriting table:', address
    except:
        print 'Creating table:', address
    
    #Upload table
    cmd = "bq --project_id=mitx-research load --skip_leading_rows=1 %s %s %s" % (address, csv_filename, schema_filename)
    print cmd
    sys.stdout.flush()
    os.system(cmd)
    os.remove(csv_filename)
    os.remove(schema_filename)


# In[1]:

def bqutil_bq2df(project_id, dataset_id, table_id):
    '''Downloads a BigQuery Table and stores it in a DataFrame.
    This method guarantees preservation of column and row orders.
    Return: A Pandas dataframe.
    '''
    info = bqutil.get_bq_table_info(dataset_id, table_id, project_id)
    megabytes = float(info['numBytes']) / 1.0e6
    print project_id + ':' + dataset_id + '.' + table_id +' is', str(megabytes) + ' Mb and', info['numRows'],'rows'
    estimated_seconds = megabytes / 0.35792536
    
    print "Downloading..."
    print "Estimated time to download table from BQ:", str(dt.timedelta(seconds = estimated_seconds))
    sys.stdout.flush()
    
    t = datetime.now()
 
    data = bqutil.get_table_data(dataset_id=dataset_id, table_id = table_id, project_id=project_id, maxResults=5000000)
    print "Download completed in", str(datetime.now() - t)
    sys.stdout.flush()
    
    if data is None:
        return pd.DataFrame()
    return pd.DataFrame(data['data'], columns=data['field_names'])


# In[3]:

def bqutil_SQL2df(project_id, SQL, temp_project_id = 'mitx-research', temp_dataset = '0_cgn_sybils', temp_table = 'temp', overwrite = True):
    '''Executes Google BigQuery SQL and stores the results in a DataFrame.
    This method guarantees preservation of column and row orders.
    Return: A Pandas dataframe.
    '''
    bqutil.create_bq_table(temp_dataset, temp_table, SQL, overwrite = overwrite, project_id = project_id, output_project_id=temp_project_id, allowLargeResults=True, sql_for_description='Created by Curtis G. Northcutt\n' + SQL)
    return bqutil_bq2df(project_id = temp_project_id, dataset_id = temp_dataset, table_id = temp_table)


# In[1]:

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6367 * c
    miles = km / 1.6
    return km


# In[1]:

def kmeans(X, n = 5, initialization = 'kmeans++',verbose = False, num_iterations = -1):
    ''' kmeans algorithm. 
        X - a nparray with rows as data points and columns as dimensions/features
        n - the number of clusters
        initialization - 'kmeans++' or 'random'
        verbose prints running times
        number_iterations - stops after num_iterations. Set to -1 for local optima without stopping.
    '''
    if initialization not in ['random','kmeans++']:
        print('Initialization must be either "random" or "kmeans++."')
              
    #initialization
    K = n #number of clusters is K
    (N, D) = np.shape(X)
    mu = np.zeros((K, D))
    r = []
    
    if initialization == 'random':
        #Randomly assign labels
        for i in range(N):
            label = np.zeros((1,K))
            label[0, random.randint(K)] = 1.0
            r.append(label)
        r = np.vstack(r)
    else:
        r = np.zeros((N, K))
        mu = kmeanspp(X, n = K, verbose = verbose)
        if verbose:
            print('Cluster means:', mu)
        #Compute closest cluster for each point
        for n in range(N):
            min_dist = np.inf
            min_k = -1
            for k in range(K):
                dist = np.linalg.norm(X[n] - mu[k])
                if dist < min_dist:
                    min_dist = dist
                    min_k = k  
            r[n, min_k] = 1.0
    
    #Iterative kmeans
    distort = []
    r_changed = True
    t = dt.datetime.now()
    iteration_count = 0
    
    while r_changed:
        
        iteration_count += 1
        r_changed = False
        
        if num_iterations != -1 and iteration_count > num_iterations:
            break
        
        #Calculate distortion
        d = distortion(r, X, mu)
        distort.append(d)
        if verbose:
            print('\nIteration', iteration_count, ', Time Elapsed:', dt.datetime.now() - t)
            print('Distortion:', d)
        
        #Compute means of clusters
        for k in range(K):
            count = np.sum(r[:, k])
            if (count != 0):
                mu[k] = np.sum(X[r[:,k] == 1.0], axis = 0) / count
        
        #Compute closest cluster for each point
        for n in range(N):
            min_dist = np.inf
            min_k = -1
            for k in range(K):
                dist = np.linalg.norm(X[n] - mu[k])
                if dist < min_dist:
                    min_dist = dist
                    min_k = k
            cur_k = np.where(r[n] == 1.0)[0][0] #Finds current cluster (i.e. index of 1 in one-hot encoded label vector)
            if (cur_k != min_k):     
                r[n] = np.zeros((1,K))
                r[n, min_k] = 1.0
                r_changed = True
            
    #Return a list of cluster labels
    return (np.array([np.where(i == 1)[0][0] for i in r]), distort)          


# In[2]:

def distortion(r, X, mu):
    '''Computes the total distortion in the clusters.
    r - responsibility one-hot encoded vectors
    X - np.array with rows as data points and columns as dimensions/features
    mu - means of clusters
    '''
    result = 0
    for n in range(len(X)):
        for k in range(len(mu)):
            result += r[n][k] * np.linalg.norm(X[n] - mu[k])
    return result


# In[3]:

def kmeanspp(X, n = 10, z = 2, verbose = False, figures = False):
    '''Runs kmeans plus plus for initialization of kmeans
    X = (N, D) matrix, input data matrix. rows are datum. colums are features/dimensions
    n = int, number of clusers.
    z = power on the distance. If z is big, more likely to select points far away from a mean.
    verbose = boolean, print out while running'''
    
    #initialization
    K = n #number of clusters is K
    (N, D) = np.shape(X)
    mu = np.zeros((K, D))
    
    #Make random datum first cluster center
    mu[0] = X[random.randint(N)] 
    d = np.zeros(N) #distance of each point to closest center
    
    if verbose:
        print('K-means++ Initialization Started.')
        k = 0
        if (figures and D == 2):
            ax = pd.DataFrame(X).plot(x = 0, y = 1, c = 'crimson', kind='scatter', figsize = (14,14), s= 500,
            title = "Kmeans++ by Curtis G. Northcutt - Iteration:" + str(k))
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            savefig('kmeanspp' + str(k) + '.png')
    
    #kmeanspp algorithm
    for k in range(1, K):
        for n in range(N):
            min_dist = np.inf
            for k_ in range(k):
                dist = np.linalg.norm(X[n] - mu[k_])
                if dist < min_dist:
                    min_dist = dist
            d[n] = min_dist
            
        #Make sure all d at least 1 by dividing by the min
        d = d / min(d[np.nonzero(d)])
        
        #Compute distribution for each point
        p = (d)**z/np.sum((d)**z) 
        
        mu[k] = X[np.random.choice(range(N), p=p)] #sample from distribution for next mean
        
        if verbose:
            print('Cluster mean', k+1, 'added.', 100.0*k/K, '% complete.')
            if (figures and D == 2):
                ax = pd.DataFrame(X).plot(x = 0, y = 1, c = 'crimson', kind='scatter', figsize = (14,14), s= 80000*p,
                title = "Kmeans++ by Curtis G. Northcutt - Iteration:" + str(k))
                ax.set_xlabel('Dimension 1')
                ax.set_ylabel('Dimension 2')
                scatter(mu[:k,0], mu[:k,1], s = 120, marker='x')
                savefig('kmeanspp' + str(k) + '.png')
                
    print('K-means++ Initialization Complete. K-means commencing.')           
    return mu


# In[1]:

def get20colors():
    '''# These are the "Tableau 20" colors as RGB.
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
 
    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)
        '''
    
    # These are the "Tableau 20" colors as RGB.
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
 
    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)
        
    return tableau20


# In[5]:

def setfont(size = 20, weight = 'normal', serif = 'Helvetica Neue', family = 'sans-serif'):
    matplotlib.rc('font', **{'family' : family,
        'serif'  : serif,
        'weight' : weight,
        'size'   : size})


# In[26]:

def getfont(family = 'sans-serif', serif = 'Helvetica Neue', weight = 'normal', size = 10):
    '''{'family' : 'sans-serif',
        'serif'  : 'Helvetica Neue',
        'weight' : 'normal',
        'size'   : 22}
        
        Call by: matplotlib.rc('font', **font)'''
    
    return {'family' : family,
        'serif'  : serif,
        'weight' : weight,
        'size'   : size}


# In[6]:

def multinomial(x):
    '''Method which does not require cumulative sum
    Implements numpy.random.choice()'''
    i = 0
    val = np.random.uniform()
    while(val >=0):
        val -= x[i]
        i += 1
    return i-1


# In[7]:

def multinomial2(x):
    '''Uses cumulative sum and finds smallest value thats great
    Implements numpy.random.choice()'''
    val = min(x[(np.cumsum(x)/max(np.cumsum(x))) > rand()])
    return np.where(x == val)[0][0]


# In[4]:

def prettyplotdf(df, title, xlabel, ylabel, kind = 'scatter', figsize = (16,10), legend = False, linewidth = 1, c = 'crimson', s = 100, cmap = cm.get_cmap('Spectral')):
    '''df - pandas DataFrame
        x will be first column, y will be next column
    '''
    ax = df.plot(x = df.columns[0], y = df.columns[1], kind = kind, figsize = figsize, legend = legend, linewidth = linewidth, c = c, title = title, s = s, cmap=cmap)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


# In[20]:

def prettyplotseries(x, title, xlabel, ylabel, kind = 'line', figsize = (16,10), legend = False, linewidth = 5, c = 'crimson', cmap = cm.get_cmap('Spectral')):
    '''x - pandas series
    '''
    if kind == 'bar':
        ax = x.plot(figsize = figsize, legend = legend, title = title, kind = kind, color = c)
    else:
        ax = x.plot(figsize = figsize, legend = legend, linewidth = linewidth, c = c,
      title = title, kind = kind, cmap = cmap)
        
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


# In[8]:

def figexamplehist(niter = 1000, dof = [1,3,5,7,9,50]):
    '''niter - number of smaples in each degree of freedom
    dof - degrees of freedom
    
    #Compute niter sqrt of sums for each of d gaussians.
    X = np.zeros((niter,6))
    for i in range(niter):
        for ix, dimension in enumerate(dof):
            X[i, ix] = np.sqrt(sum(np.random.normal(size = dimension)**2))

    #Plot the histograms
    plt.figure(figsize = (16, 10))
    for i, dimension in enumerate(dof):
        plt.hist(X[:,i], alpha = .4, bins = 50, linewidth = .1, label = 'dimensions (df) =' +str(dimension))

    plt.title('Sqrt of Squared Sum of "d" Independent Normal Distributions', fontsize=20)
    plt.xlabel('Euclidean Distance from Origin', fontsize=18)
    plt.ylabel('Histogram of', niter, 'samples for each dimension', fontsize=16)
    legendcontent = ['d = ' + str(d) for d in dof]
    plt.legend([legendcontent, title = 'Number of Dimensions (Independent Guassians in Sum)')
    '''
    #Compute niter sqrt of sums for each of d gaussians.
    X = np.zeros((niter,len(dof)))
    for i in range(niter):
        for ix, dimension in enumerate(dof):
            X[i, ix] = np.sqrt(sum(np.random.normal(size = dimension)**2))

    #Plot the histograms
    plt.figure(figsize = (16, 10))
    for i, dimension in enumerate(dof):
        plt.hist(X[:,i], alpha = .4, bins = 50, linewidth = .1, label = 'dimensions (df) =' +str(dimension))

    plt.title('Sqrt of Squared Sum of "d" Independent Normal Distributions', fontsize=20)
    plt.xlabel('Euclidean Distance from Origin', fontsize=18)
    plt.ylabel('Histogram of '+ str(niter) + ' samples for each dimension', fontsize=16)
    legendcontent = ['d = ' + str(d) for d in dof]
    plt.legend(legendcontent, title = 'Number of Dimensions (Independent Guassians in Sum)')


# In[24]:

def figexamplemultiscatter(niter = 1000, dof = [1,3,5,7,9,50]):
    '''
    plt.figure(figsize = (14, 8))
    dimension = 1
    x = np.linspace(0, 10, 1000)
    for dimension in dof:
        y = [scipy.stats.chi.pdf(i, df = dimension) for i in x]
        plt.plot(x,y)
    plt.title('Chi Distribution', fontsize=20)
    plt.xlabel('Random Variable d, Distance from Origin', fontsize=18)
    plt.ylabel('Probability Density Function', fontsize=16)
    legendcontent = ['df = ' + str(d) for d in dof]
    plt.legend(legendcontent, title = 'Number of Dimensions (degrees of freedom for Chi)')
   '''
    plt.figure(figsize = (14, 8))
    dimension = 1
    x = np.linspace(0, 20, 1000)
    for dimension in dof:
        y = [scipy.stats.chi.pdf(i, df = dimension) for i in x]
        plt.plot(x,y)
    plt.title('Chi Distribution', fontsize=20)
    plt.xlabel('Random Variable d, Distance from Origin', fontsize=18)
    plt.ylabel('Probability Density Function', fontsize=16)
    legendcontent = ['df = ' + str(d) for d in dof]
    plt.legend(legendcontent, title = 'Number of Dimensions (degrees of freedom for Chi)') 


# In[2]:

def generateRandomClusters(n = 100, seperation = 6, uniformity = 1, gridSize = 5, randomness = 1.5):
    '''
    seperation - how distinct the grid clusters are (higher value is more sperated)
    uniformity - lower uniformity tends to cluster at (0,0), higher uniformity makes all grid spots equally likely
    gridSize - (gridSize x gridSize) lattice where clusters can go.
    randomness - Variation within each cluster (exhibits uniform distribution as randomness grows larger)
    
    Explanation of how this works: randint(randint(3) + 1)*5 + np.random.normal()
    The innermost term,randint(3) + 1, generates a random max for randint from .
    randint then generates some random number between 0 and the random number we just found.
    This is multiplied by 5, this spaces out the results of the random. For example,
    if you generate random numbers in range [0,1,2] --> x5 --> [0,5, 10] (better seperate clusters)
    Finally, we add gaussian noise around this random location. 
    In summary, we have a bunch of points we could generate centers, and how many times
    we generate points in each point is randomized. And then we also add random noise to each
    point. This results in light clustering.'''
    
    gridSize -= uniformity
    X = [[np.random.randint(np.random.randint(gridSize) + uniformity)*seperation + randomness*np.random.normal(), 
          np.random.randint(np.random.randint(gridSize) + uniformity)*seperation + randomness*np.random.normal()] 
         for i in range(n)]
    return np.array(X)

#pd.DataFrame(X).plot(x = 0, y = 1, kind = 'scatter', figsize=(14,8), s = 50)


# In[ ]:

def countrymap(country = None):
    cmap = buildcountrymap()
    if country == None:
        return cmap
    else:
        return cmap[country]


# In[66]:

def buildcountrymap():
    locmap = {unicodedata.normalize('NFKD', i.name).encode('ascii','ignore').decode('utf-8') : 
        int(i.numeric) for i in list(pycountry.countries)}
    locmap['Venezuela'] = locmap['Venezuela, Bolivarian Republic of']
    locmap['Bolivia'] = locmap['Bolivia, Plurinational State of']
    locmap['Macedonia'] = locmap['Macedonia, Republic of']
    locmap['Libyan Arab Jamahiriya'] = locmap['Libya']
    locmap['Taiwan'] = locmap['Taiwan, Province of China']
    locmap['Moldova'] = locmap['Moldova, Republic of']
    locmap['Antigua'] = locmap['Antigua and Barbuda']
    locmap['Barbuda'] = locmap['Antigua and Barbuda']
    locmap['Bonaire'] = locmap['Bonaire, Sint Eustatius and Saba']
    locmap['Bosnia'] = locmap['Bosnia and Herzegovina']
    locmap['Herzegovina'] = locmap['Bosnia and Herzegovina']
    locmap['Cocos Islands'] = locmap['Cocos (Keeling) Islands']
    locmap['Congo'] = locmap['Congo, The Democratic Republic of the']
    locmap['Falkland Islands'] = locmap['Falkland Islands (Malvinas)']
    locmap['Iran'] = locmap['Iran, Islamic Republic of']
    locmap['Holy See'] = locmap['Holy See (Vatican City State)']
    locmap['Vatican City State'] = locmap['Holy See (Vatican City State)']
    locmap['Heard Island'] = locmap['Heard Island and McDonald Islands']
    locmap['McDonald Islands'] = locmap['Heard Island and McDonald Islands']
    locmap['Korea'] = locmap['Korea, Democratic People\'s Republic of']
    locmap['Lao'] = locmap['Lao People\'s Democratic Republic']
    locmap['Micronesia'] = locmap['Micronesia, Federated States of']
    locmap['Palestine'] = locmap['Palestine, State of']
    locmap['Saint Helena'] = locmap['Saint Helena, Ascension and Tristan da Cunha']
    locmap['Saint Martin'] = locmap['Saint Martin (French part)']
    locmap['Saint Kitts'] = locmap['Saint Kitts and Nevis']
    locmap['Nevis'] = locmap['Saint Kitts and Nevis']
    locmap['Saint Pierre'] = locmap['Saint Pierre and Miquelon']
    locmap['Miquelon'] = locmap['Saint Pierre and Miquelon']
    locmap['Saint Vincent'] = locmap['Saint Vincent and the Grenadines']
    locmap['Grenadines'] = locmap['Saint Vincent and the Grenadines']
    locmap['Sao Tome'] = locmap['Sao Tome and Principe']
    locmap['Principe'] = locmap['Sao Tome and Principe']
    locmap['Sint Maarten'] = locmap['Sint Maarten (Dutch part)']
    locmap['South Georgia'] = locmap['South Georgia and the South Sandwich Islands']
    locmap['South Sandwich Islands'] = locmap['South Georgia and the South Sandwich Islands']
    locmap['Svalbard'] = locmap['Svalbard and Jan Mayen']
    locmap['Jan Mayen'] = locmap['Svalbard and Jan Mayen']
    locmap['Tanzania'] = locmap['Tanzania, United Republic of']
    locmap['Trinidad'] = locmap['Trinidad and Tobago']
    locmap['Tobago'] = locmap['Trinidad and Tobago']
    locmap['Turks'] = locmap['Turks and Caicos Islands']
    locmap['Caicos Islands'] = locmap['Turks and Caicos Islands']
    locmap['Wallis'] = locmap['Wallis and Futuna']
    locmap['Futuna'] = locmap['Wallis and Futuna']
    locmap['British Virgin Islands'] = locmap['Virgin Islands, British']
    locmap['U.S. Virgin Islands'] = locmap['Virgin Islands, U.S.']
    locmap['Virgin Islands'] = locmap['Virgin Islands, U.S.']
    locmap['Heard Island and Mcdonald Islands'] = locmap['Heard Island and McDonald Islands']
    locmap['Netherlands Antilles'] = locmap['Netherlands']
    locmap['Palestinian Territory, Occupied'] = locmap['Palestine']
    locmap['Congo, the Democratic Republic of the'] = locmap['Congo']
    locmap['Cote D\'Ivoire'] = locmap['Cote d\'Ivoire']
    locmap['Virgin Islands, U.s.'] = locmap['Virgin Islands, U.S.']
    return locmap


# In[2]:

def prettyhistpercent(
    series,
    title = 'Histogram of percent of total questions answered', 
    xlabel = 'X = Percent of total questions answered',
    ylabel= 'Number of students who answered X percent of questions',
    bins = 40,
    titlefontsize = 20,
    axisfontsize = 16,
    figsize = (16,10),
    legend = False,
    linewidth = .1,
    c = 'crimson',
    s = 100,
    alpha = .5,
    legendloc = 2,
    legendcontent = 'Content of legend',
    legendtitle = 'About the data'):
    
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(1,1,1)
    plt.hist(series, alpha = alpha, bins = bins, linewidth = linewidth, color = c)
    plt.title(title, fontsize=titlefontsize)
    plt.xlabel(xlabel, fontsize=axisfontsize)
    plt.ylabel(ylabel, fontsize=axisfontsize)
    plt.legend(legendcontent, title = legendtitle, loc = legendloc)
    fmt = '%.f%%' # Format you want the ticks, e.g. '40%'
    xticks = mtick.FormatStrFormatter(fmt)
    ax.xaxis.set_major_formatter(xticks)
    #plt.savefig('chinax_percent_of_questions_answered_by_possible_cheaters.png')


# In[ ]:



