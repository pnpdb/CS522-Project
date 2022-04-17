from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
import pandas as pd
class SVD:
    def __init__(self, r):
        self.R = r
        self.SVD = None
        self.U = None
        self.VT = None
        self.V = None
        pass
    
    def Process(self, mat):
        self.SVD = TruncatedSVD(self.R, algorithm = 'arpack')
        self.U = self.SVD.fit_transform(mat)
        self.U = Normalizer(copy=False).fit_transform(self.U)
        self.U = pd.DataFrame(self.U)
        
        self.VT = pd.DataFrame(self.SVD.components_)
        self.V = self.VT.T
    
    