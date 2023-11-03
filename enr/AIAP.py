import torch
import numpy as np
from enr.DDG import *
from enr.varifold import *
from enr.regularizers import *
from torch.autograd import grad
from pykeops.torch import kernel_product, Genred
from pykeops.torch.kernel_product.formula import *

use_cuda = 1
torchdeviceId = torch.device('cuda:0') if use_cuda else 'cpu'
torchdtype = torch.float32

##############################################################################################################################
#H2 Helper Functions
##############################################################################################################################


def getPathEnergyAIAP(geod,a,F_sol,stepwise=False):
    N=geod.shape[0]        
    diff=(geod[1:,:,:] - geod[:-1,:,:])*N
    enr=0
    step_enr=torch.zeros((N-1,1),dtype=torchdtype)   
    K=getAIAPenergy()
    for i in range(0,N-1):   
        dv=diff[i]        
        
        enr+=K(diff[i],geod[i])/N
        
        if a>0:
            M=getVertAreas(geod[i],F_sol)
            Ndv=M*batchDot(dv,dv)
            enr+=a*torch.sum(Ndv)/N    
            
        if stepwise:
            if i==0:
                step_enr[0]=enr
            else:
                step_enr[i]=enr-torch.sum(step_enr[0:i])
    if stepwise:
        return enr,step_enr    
    return enr

def getAIAPenergy():
    def K(Xp, p):
        d = Xp.shape[1]
        pK = Genred('((X-Y)|(p-q))*((X-Y)|(p-q))',
            ['X=Vi('+str(d)+')','Y=Vj('+str(d)+')','p=Vi('+str(d)+')',
            'q=Vj('+str(d)+')'],reduction_op='Sum',axis=1)
        M=pK(Xp,Xp,p,p)
        return (torch.dot(M.view(-1), torch.ones_like(M).view(-1))).sum()
    return K        


##############################################################################################################################
#H2_Matching_Energies
##############################################################################################################################

def enr_match_AIAP_sym(VS, FS, FunS, VT, FT, FunT, F_sol, Fun_sol, B_sol, weight_coef_dist_S=1 ,weight_coef_dist_T=1, weight_Gab = 1,a=1, **objfun):

    K = VKerenl(objfun['kernel_geom'],objfun['kernel_grass'],objfun['kernel_fun'],objfun['sig_geom'],objfun['sig_grass'],objfun['sig_fun'])
    dataloss_S = lossVarifoldSurf(F_sol, Fun_sol, VS, FS, FunS, K)
    dataloss_T = lossVarifoldSurf(F_sol, Fun_sol, VT, FT, FunT, K)
    
    
    def energy(geod):
        enr=getPathEnergyAIAP(geod,a,F_sol)
        N=geod.shape[0]    
        E=weight_Gab*enr + weight_coef_dist_S*dataloss_S(geod[0]) + weight_coef_dist_T*dataloss_T(geod[N-1])
        return E
    return energy

def enr_match_AIAP(VS, VT, FT, FunT, F_sol, Fun_sol, B_sol, weight_coef_dist_T=1, weight_Gab=1,a=1, **objfun):

    K = VKerenl(objfun['kernel_geom'],objfun['kernel_grass'],objfun['kernel_fun'],objfun['sig_geom'],objfun['sig_grass'],objfun['sig_fun'])
    dataloss_T = lossVarifoldSurf(F_sol, Fun_sol, VT, FT, FunT, K)   
    def energy(geod):
        #for i in geod.shape[0]:
        #    L=getLaplacian(geod[i],F_sol)
        #    geod[i]=geod[i]+L(geod[i])
        #    geod[i]=geod[i]-L(geod[i])
        geod=torch.cat((torch.unsqueeze(VS,dim=0),geod),dim=0).to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)
        enr=getPathEnergyAIAP(geod,a,F_sol)
        N=geod.shape[0]    
        E=weight_Gab*enr + weight_coef_dist_T*dataloss_T(geod[N-1])
        return E
    return energy


def enr_match_AIAP_sym_w(VS, FS, FunS, VT, FT, FunT, F_sol, Fun_sol, Rho, B_sol, weight_coef_dist_S=1 ,weight_coef_dist_T=1, weight_Gab = 1,a=1, **objfun):

    K = VKerenl(objfun['kernel_geom'],objfun['kernel_grass'],objfun['kernel_fun'],objfun['sig_geom'],objfun['sig_grass'],objfun['sig_fun'])
    
    
    dataloss_S = lossVarifoldSurf(F_sol, Fun_sol,VS, FS, FunS, K)
    dataloss_T = lossVarifoldSurf_Weighted(F_sol, Fun_sol, VT, FT, FunT, K)
    
    
    def energy(geod,Rho):
        enr=getPathEnergyH2(geod,a,F_sol)
        N=geod.shape[0]    
        E=weight_Gab*enr + weight_coef_dist_S*dataloss_S(geod[0]) + weight_coef_dist_T*dataloss_T(geod[N-1],torch.clamp(Rho,-.25,1.25))#+.01*penalty(geod[N-1],F_sol, Rho)
        return E
    return energy

def enr_param_AIAP(left,right, F_sol,a):    
    def energy(mid):
        geod=torch.cat((left, mid,right),dim=0).to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)
        enr=getPathEnergyAIAP(geod,a,F_sol)        
        return enr 
    return energy


def enr_match_AIAP_coeff(VS, VT, FT, FunT, F_sol, Fun_sol, geod, basis, weight_coef_dist_T=1, weight_Gab=1, a=1, **objfun):
    K = VKerenl(objfun['kernel_geom'], objfun['kernel_grass'], objfun['kernel_fun'], objfun['sig_geom'],
                objfun['sig_grass'], objfun['sig_fun'])
    dataloss_T = lossVarifoldSurf(F_sol, Fun_sol, VT, FT, FunT, K)

    def energy(X):
        X=torch.cat((torch.unsqueeze(torch.zeros((X.shape[1])).to(dtype=torchdtype, device=torchdeviceId),dim=0),X),dim=0).requires_grad_(True)
        geod_torch = geod + torch.einsum("ij, jkl-> ikl", X, basis)
        enr = getPathEnergyAIAP(geod_torch, a, F_sol)
        
        N = geod.shape[0]
        var2 = weight_coef_dist_T * dataloss_T(geod_torch[N - 1])
        E = weight_Gab * enr + var2
        return E
    return energy

def enr_match_AIAP_sym_coeff(VS, FS, FunS, VT, FT, FunT, F_sol, Fun_sol, geod, basis,
                         weight_coef_dist_S=1, weight_coef_dist_T=1, weight_Gab=1, a=1, **objfun):
    K = VKerenl(objfun['kernel_geom'], objfun['kernel_grass'], objfun['kernel_fun'], objfun['sig_geom'],
                objfun['sig_grass'], objfun['sig_fun'])
    dataloss_S = lossVarifoldSurf(F_sol, Fun_sol, VS, FS, FunS, K)
    dataloss_T = lossVarifoldSurf(F_sol, Fun_sol, VT, FT, FunT, K)

    def energy(X):
        geod_torch = geod + torch.einsum("ij, jkl-> ikl", X, basis)
        enr = getPathEnergyAIAP(geod_torch, a, F_sol)
        
        N = geod.shape[0]
        var1 = weight_coef_dist_S * dataloss_S(geod_torch[0])
        var2 = weight_coef_dist_T * dataloss_T(geod_torch[N - 1])
        E = weight_Gab * enr + var1 + var2
        return E

    return energy

def enr_param_AIAP_coeff( F_sol, geod, basis, a=1, **objfun):
    
    def energy(X):
        X=torch.cat((torch.unsqueeze(torch.zeros((X.shape[1])).to(dtype=torchdtype, device=torchdeviceId),dim=0),X,torch.unsqueeze(torch.zeros((X.shape[1])).to(dtype=torchdtype, device=torchdeviceId),dim=0)),dim=0).requires_grad_(True)
        geod_torch = geod + torch.einsum("ij, jkl-> ikl", X, basis)
        return getPathEnergyAIAP(geod_torch, a, F_sol)

    return energy