import numpy as np
class LogObj:
    '''
    The below is a pythonic version of the Loglet Lab 4 code found at https://github.com/pheguest/logletlab4. 
    which allows you to create amazing graphs. However, the issue with the web interface is that it is not 
    designed for processing hundreds of time series, and in order to do this, each time series must be 
    laboriously copy-pasted into the input box, the parameters set, and then the results saved individually. 
    With 200 time series and multiple parameter sets, this process is quite slow! Therefore, we have adapted 
    the code from the github repository, but the original should be seen at 
    https://github.com/pheguest/logletlab4/blob/master/javascript/src/psmlogfunc3.js. 
    '''


    def __init__(self, x, y, number_of_loglets):
        self.data = {
            'x': x,
            'y': y,
            'yfit': [],
            'residuals': [],
            'standardised_residuals': [],
            'number_of_points': []
        }
        self.curve = {
            'number_of_points': 256,
            't': [],
            'yfit': []
        }
        self.flot = {
            'data': [],
            'curve': [],
            'residuals': [],
            'standardised_residuals': [],
            'resline': []
        }
        self.fp = []
        self.cc = []
        self.MCiter = 5000
        self.anneal = 25
        self.anneal_iter = 10
        self.number_of_loglets = number_of_loglets
        self.energy_best = 1e23
        self.parameters = {
            'd': 0,
            'a': np.ones(self.number_of_loglets),
            'k': np.ones(self.number_of_loglets),
            'b': np.ones(self.number_of_loglets)
        }
        
def randnum(low,high):
    return (high - low)*np.random.random() + low

def loglet(t, a, k, b):
    # log_e(81) = 4.394449154672439 
    return k/(1+np.exp((t-b)*(4.394449154672439/(-a))))

def gompertz(t, a, k, b):
    # log_e(log_e(81)) = 1.4803421887365897
    return k*np.exp(-np.exp((1.4803421887365897/(-a))*(t-b)))

def estimate_constraints(x, y, number_of_loglets):
    # We take the constraints we use from Loglet Lab's online demo at https://logletlab.com
    constraints = {
        'dlow': 0,
        'dhigh': 0
    }
    for i in range(number_of_loglets):
        constraints['klow'+str(i)] = (1/(number_of_loglets))*0.5*max(y)
        constraints['khigh'+str(i)] = (1/(number_of_loglets))*4*max(y)
        
        constraints['alow'+str(i)] = (1/(number_of_loglets))*(max(x)-min(x))/8
        constraints['ahigh'+str(i)] = (1/(number_of_loglets))*(max(x)-min(x))
        
    if number_of_loglets == 1:
        constraints['blow0'] = min(x)
        constraints['bhigh0'] = max(x)*2 - min(x)
        
    elif number_of_loglets == 2:
        constraints['blow0'] = min(x)
        constraints['bhigh0'] = ((max(x) - min(x))/2) + min(x)
        constraints['blow1'] = ((max(x) - min(x))/2) + min(x)
        constraints['bhigh1'] = ((max(x) - min(x))/2) + max(x)
    
    return constraints


def loglet_MC_anneal_regression(logobj, constraints=None, number_of_loglets=1, curve_type='logistic', anneal_iterations=50, mc_iterations=5000, anneal_sample_size=100):
    '''
    From the original description of the function: 
    /* this function uses a simple Monte-Carlo based */
    /* "simulated annealing" algorithm to perform    */
    /* constrained non-linear regression.            */
    /* see http://lizardinthesun.com/cloudblog/      */
    /* for details. 
    '''
    
    if constraints == None:
        constraints = estimate_constraints(logobj.data['x'], logobj.data['y'], number_of_loglets)

    # Set the simulated annealing parameters
    T_min = 20
    alpha = (T_min/anneal_sample_size)**(1/anneal_iterations)

    
    # Set up initial conditions
    point_d = 0
    point_a = np.ones(number_of_loglets)
    point_k = np.ones(number_of_loglets)
    point_b = np.ones(number_of_loglets)
    
    y_try = np.zeros(len(logobj.data['x']))
    iteration = 0
    energy = 0
    
    n = mc_iterations
    
    sample_d = np.zeros(n)
    sample_a = np.zeros([number_of_loglets, n])
    sample_k = np.zeros([number_of_loglets, n])
    sample_b = np.zeros([number_of_loglets, n])
    
    anneal_d = np.zeros(n)
    anneal_a = np.zeros([number_of_loglets, n])
    anneal_k = np.zeros([number_of_loglets, n])
    anneal_b = np.zeros([number_of_loglets, n])
    
    # first argument is energy, second is index
    sample_energy = {}
    
    anneal = {}
    for i in range(number_of_loglets):
        if i==0:
            anneal["min_d"] = constraints["dlow"]
            anneal["max_d"] = constraints["dhigh"]
        anneal["min_a"+str(i)] = constraints["alow"+str(i)];
        anneal["max_a"+str(i)] = constraints["ahigh"+str(i)];
        anneal["min_k"+str(i)] = constraints["klow"+str(i)];
        anneal["max_k"+str(i)] = constraints["khigh"+str(i)];
        anneal["min_b"+str(i)] = constraints["blow"+str(i)];
        anneal["max_b"+str(i)] = constraints["bhigh"+str(i)];  
        

    # Start the annealing process
    
    for aiter in range(anneal_iterations):

        for mciter in range(mc_iterations):
            for i in range(number_of_loglets):
                if i==0:
                    point_d = randnum(anneal["min_d"], anneal["max_d"])
                point_a[i] = randnum(anneal["min_a"+str(i)], anneal["max_a"+str(i)])
                point_k[i] = randnum(anneal["min_k"+str(i)], anneal["max_k"+str(i)])
                point_b[i] = randnum(anneal["min_b"+str(i)], anneal["max_b"+str(i)])

            y_try = np.zeros(len(logobj.data['x']))
            for j in range(len(logobj.data['x'])):
                for i in range(number_of_loglets):
                    if curve_type == 'logistic':
                        y_try[j] = y_try[j] + loglet(logobj.data['x'][j], point_a[i], point_k[i], point_b[i]);
                    elif curve_type == 'gompertz':
                        y_try[j] = y_try[j] + gompertz(logobj.data['x'][j], point_a[i], point_k[i], point_b[i]);
                    if i == 0:
                        y_try[j] += point_d
                        
            for i in range(number_of_loglets):
                if i == 0:
                    sample_d[mciter] = point_d
                sample_a[i][mciter] = point_a[i]
                sample_k[i][mciter] = point_k[i]
                sample_b[i][mciter] = point_b[i]
                
            energy = np.sqrt(((logobj.data['y'] - y_try)**2).mean())
            
            sample_energy[mciter] = energy
            
            if energy < logobj.energy_best:
                for i in range(number_of_loglets):
                    if i==0:
                        logobj.parameters['d'] = point_d
                    logobj.parameters['a'][i] = point_a[i]
                    logobj.parameters['k'][i] = point_k[i]
                    logobj.parameters['b'][i] = point_b[i]

                logobj.energy_best = energy
                
                
        # Sort, in order to contract parameter constraints (simulated annealing)
        # My notes: we sort the samples in order of their energy
        # Sort so that lowest energy comes first
        sorted_energy_indices = sorted(sample_energy, key=lambda k: sample_energy[k])

        for i in range(number_of_loglets):
            if i==0:
                for j in range(n):
                    anneal_d[j] = sample_d[sorted_energy_indices[j]]
                anneal["min_d"] = anneal_d[0:anneal_sample_size].min()
                anneal["max_d"] = anneal_d[0:anneal_sample_size].max()
                
                # make certain that the best point is not excluded
                if anneal["min_d"] > logobj.parameters['d']:
                    anneal["min_d"] = logobj.parameters['d']
                if anneal["max_d"] < logobj.parameters['d']:
                    anneal["max_d"] = logobj.parameters['d']
                    
            for j in range(n):
                anneal_a[i][j] = sample_a[i][sorted_energy_indices[j]]
            anneal["min_a"+str(i)] = anneal_a[i][0:anneal_sample_size].min();
            anneal["max_a"+str(i)] = anneal_a[i][0:anneal_sample_size].max();
            
            # Make certain that the best point is not accidentally excluded
            if anneal["min_a"+str(i)] > logobj.parameters['a'][i]:
                anneal["min_a"+str(i)] = logobj.parameters['a'][i]
            if anneal["max_a"+str(i)] < logobj.parameters['a'][i]:
                anneal["max_a"+str(i)] = logobj.parameters['a'][i]

            for j in range(n):
                anneal_k[i][j] = sample_k[i][sorted_energy_indices[j]]
            anneal["min_k"+str(i)] = anneal_k[i][0:anneal_sample_size].min();
            anneal["max_k"+str(i)] = anneal_k[i][0:anneal_sample_size].max();

            # Make certain that the best point is not accidentally excluded
            if anneal["min_k"+str(i)] > logobj.parameters['k'][i]:
                anneal["min_k"+str(i)] = logobj.parameters['k'][i]
            if anneal["max_k"+str(i)] < logobj.parameters['k'][i]:
                anneal["max_k"+str(i)] = logobj.parameters['k'][i]
                
            for j in range(n):
                anneal_b[i][j] = sample_b[i][sorted_energy_indices[j]]
            anneal["min_b"+str(i)] = anneal_b[i][0:anneal_sample_size].min();
            anneal["max_b"+str(i)] = anneal_b[i][0:anneal_sample_size].max();
            
            # Make certain that the best point is not accidentally excluded
            if anneal["min_b"+str(i)] > logobj.parameters['b'][i]:
                anneal["min_b"+str(i)] = logobj.parameters['b'][i]
            if anneal["max_b"+str(i)] < logobj.parameters['b'][i]:
                anneal["max_b"+str(i)] = logobj.parameters['b'][i]
            
        anneal_sample_size = int(anneal_sample_size*alpha)
    return logobj
            
    
def calculate_series(x, a, k, b, curve_type):
    '''
    Given a set of curve parameters, return a fitted curve
    '''
    y_try = np.zeros(len(x))
    for j in range(len(x)):
        if curve_type=='logistic':
            y_try[j] = loglet(x[j], a, k, b)
        elif curve_type=='gompertz':
            y_try[j] = gompertz(x[j], a, k, b)
        
    return y_try


def calculate_series_double(x, a1, k1, b1, a2, k2, b2, curve_type):
    '''
    Outputs a list of three lists
    '''
    y_1 = np.zeros(len(x))
    y_2 = np.zeros(len(x))
    
    for j in range(len(x)):
        if curve_type=='logistic':
            y_1[j] = loglet(x[j], a1, k1, b1)
            y_2[j] = loglet(x[j], a2, k2, b2)
        elif curve_type=='gompertz':
            y_1[j] = gompertz(x[j], a1, k1, b1)
            y_2[j] = gompertz(x[j], a2, k2, b2)
        
    return y_1+y_2, y_1, y_2