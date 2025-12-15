# -*- coding: utf-8 -*-


 
import numpy as np
import math
from hmmlearn.hmm import GaussianHMM
from typing import List, Tuple, Iterator

from scipy.stats import rv_continuous
 
 
from scipy.stats import norm
from scipy.special import logsumexp


class HMM:
    def __init__(self, n_states: int, n_dimensions: int, random_seed: int = None):
        """
        Crée un HMM avec des paramètres générés aléatoirement.
        
        Args:
            n_states (int): Nombre d'états cachés.
            n_dimensions (int): Dimension des observations (pour les émissions gaussiennes).
            random_seed (int, optional): Graine pour la reproductibilité.
        """
        self.n_states = n_states
        self.n_dimensions = n_dimensions
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialiser les paramètres du HMM
        self.startprob = None  # Probabilités initiales (pi)
        self.transmat = None   # Matrice de transition (A)
        self.means = None      # Moyennes des gaussiennes
        self.covars = None     # Matrices de covariance des gaussiennes
        self._initialize_hmm()

    
    def _initialize_hmm(self):
        """
        Initialise les paramètres du HMM aléatoirement.
        """
        # Probabilités initiales (pi) via Dirichlet
        self.startprob = np.random.dirichlet(np.ones(self.n_states))
        
        # Matrice de transition (A) via Dirichlet pour chaque ligne
        self.transmat = np.random.dirichlet(np.ones(self.n_states), size=self.n_states)
        
        # Moyennes des distributions d'émission (gaussiennes)
        self.means = np.random.randn(self.n_states, self.n_dimensions) * 2  # Écart-type arbitraire
        
        # Matrices de covariance (positives définies)
        self.covars = np.zeros((self.n_states, self.n_dimensions, self.n_dimensions))
        for i in range(self.n_states):
            A = np.random.randn(self.n_dimensions, self.n_dimensions)
            self.covars[i] = np.dot(A, A.T) + np.eye(self.n_dimensions) * 0.1  # Stabilité



class VHEM:
    def __init__(self, n_reduced: int, n_base: int,
                 max_iterations: int = 100, tol: float = 1e-4):
        
        
        self.n_reduced = n_reduced
        self.n_base = n_base
        
        self.max_iterations = max_iterations
        self.tol = tol
        
        self.base_hmms = []
        self.reduced_hmms = []
             
        self.weights = None     # Poids des clusters (mélange)
        self.base_weights = None
        
    def initialize(self):
        """Initialise les HMMs des clusters et les poids."""
        
        
        for i in range(self.n_base):
            local = HMM(4, 2)
            local.initialize()
            self.base_hmms.append(local)
         
        
        for i in range(self.n_reduced):
            local = HMM(4, 2)
            local.initialize()
            self.reduced_hmms.append(local)  
        
        self.reduced_weights = np.ones(self.n_reduced) / self.n_reduced
        self.base_weights = np.ones(self.n_base) / self.n_base
        # Initialisation aléatoire des HMMs de clusters
        #indices = np.random.choice(n_hmms, self.n_clusters, replace=False)
        #self.cluster_hmms = [hmms[i] for i in indices]
        
        
        
    def process(self):
        
        eta = np.zeros((i,j))
        for i in range(self.n_base):
            for j in range (self.n_reduced):
                num = 
                
                
        
        




VHEM(n_reduced=3, n_base = 10)















