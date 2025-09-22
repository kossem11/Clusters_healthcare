import numpy as np

class SubClass:
    def __init__(self, settlist):
        self.settlist = settlist

    def relSVD(seld, A_mtx, B_vect):
        U , S , Vt = np.linalg.svd(A_mtx, full_matrices=False)
        #Sing = np.diag(S)
        Sing_plus = np.diag(1/S)
        A_rev = Vt.T @ Sing_plus @ U.T
        x = A_rev @ B_vect
        res_miss = B_vect - (A_mtx @ x)
        return x, res_miss
    
    def onRnd(self):
        _,len,height,_ = self.settlist
        A = np.random.rand(len, height)
        B = np.random.rand(height)
        return A, B
