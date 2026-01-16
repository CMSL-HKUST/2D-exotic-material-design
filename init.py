from fenics import *
import numpy as np
from definition import*


def init(case):
    if case =='Horizontal rigidity maximization':
        Nx,Ny,kappa0_initial,delta,delta2,lam,gamma,threshold=[20,20, 0.5, 0.8, 0.8, 10, 1e-2,-0.02] #threshold =-0.02
        lsf_init=Expression('1/N*(pow(cos(pi*(x[0]-0.5)),2)*pow(cos(pi*(x[1]-0.5)),2)-0.5)',N=1,degree=2)
    if case =='Bulk modulus maximization':
        Nx,Ny,kappa0_initial,delta,delta2,lam,gamma,threshold=[40,40, 0.5, 0.8, 0.8, 240, 1e-2,-0.05] 
        lsf_init=Expression( '-1/N*(pow(cos(pi*(x[0]-0.5)),2)*pow(cos(pi*(x[1]-0.5)),2)-0.5)*(pow(cos(pi*x[0]),2)*pow(cos(pi*(x[1]-1)),2)-0.9)',N=1,degree=2)
    if case =='Shear modulus maximization':
        Nx,Ny,kappa0_initial,delta,delta2,lam,gamma,threshold=[40,40, 0.5, 0.8, 0.8, 50, 1e-2,-0.18] 
        lsf_init=Expression('1/N*(pow(cos(pi*(x[0]-0.5)),2)*pow(cos(pi*(x[1]-0.5)),2)-0.6)',N=1,degree=2)
    if case =='minimization of modified Poisson ratio':
        Nx,Ny,kappa0_initial,delta,delta2,lam,gamma,threshold=[60,60, 0.08, 0.8, 1, 0.5, 1e-2,-0.05] 
        lsf_init=Expression('1/N*(pow(cos(pi*(x[0]-0.5)),2)*pow(cos(pi*(x[1]-0.5)),2)-0.5)',N=1,degree=2)
    if case =='maximization of Poisson ratio':
        Nx,Ny,kappa0_initial,delta,delta2,lam,gamma,threshold=[60,60, 0.1, 0.8, 0.8, 0.5, 1e-2,0] 
        lsf_init=Expression('1/N*(pow(cos(pi*(x[0]-0.5)),2)*pow(cos(pi*(x[1]-0.5)),2)-0.5)',N=1,degree=2)
    if case =='R0 orthotrope':
        Nx,Ny,kappa0_initial,delta,delta2,lam,gamma,threshold=[120,120, 0.2, 0.8, 0.8, 0, 1e-3,-0.05] 
        lsf_init=Expression('1/N*(pow(cos(0.55*pi*(x[0]-0.5)),2)*pow(cos(0.3*pi*(x[1]-0.5)),2)-0.95)',N=1,degree=2)
    if case =='R0 orthotrope_plus':
        Nx,Ny,kappa0_initial,delta,delta2,lam,gamma,threshold=[120,120, 0.2, 0.8, 0.8, 0, 1e-3,-0.05] 
        lsf_init=Expression('1/N*(pow(cos(0.55*pi*(x[0]-0.5)),2)*pow(cos(0.3*pi*(x[1]-0.5)),2)-0.95)',N=1,degree=2)
    if case=='Cauchy elasticity':
        Nx,Ny,kappa0_initial,delta,delta2,lam,gamma,threshold=[80,80, 0.1, 0.8, 0.9, 0, 1e-3, 0.05]  #[60,60, 0.1, 0.8, 0.98, 1, 1e-3,0] 
        lsf_init=Expression('1/N*(pow(cos(pi*(x[0]-0.5)),2)*pow(cos(pi*(x[1]-0.5)),2)-0.5)',N=1,degree=2)
    if case=='minimisation of Poisson ration under contraint on E':
        Nx,Ny,kappa0_initial,delta,delta2,lam,gamma,threshold=[40,40, 0.05, 0.8, 0.95, 1, 1e-2,0] 
        lsf_init=Expression('1/N*(pow(cos(pi*(x[0]-0.5)),2)*pow(cos(pi*(x[1]-0.5)),2)-0.5)',N=1,degree=2)
    return Nx,Ny,kappa0_initial,delta,delta2,lam,gamma,threshold,lsf_init

# -1/N*(pow(cos(pi*(x[0]-0.5)),2)*pow(cos(pi*(x[1]-0.5)),2)-0.4)*(pow(cos(pi*x[0]),2)*pow(cos(pi*(x[1]-1)),2)-0.9)
#'1/N*(pow(cos(5*pi*(x[0]-0.1)),2)*pow(cos(5*pi*(x[1]-0.1)),2)-0.65)'
def init_type(type,case,Chom,D_TC):
    if type=='Type1':
        if case=='Horizontal rigidity maximization':
            h=Type1_h(Chom,0,0,0,0)
            D_TJ=Type1_DTJ(Chom,D_TC,0,0,0,0)
        if case =='Bulk modulus maximization':
            h=Type1_h(Chom,0,0,0,0)+2*Type1_h(Chom,0,0,1,1)+Type1_h(Chom,1,1,1,1)
            D_TJ=Type1_DTJ(Chom,D_TC,0,0,0,0)+2*Type1_DTJ(Chom,D_TC,0,0,1,1)+Type1_DTJ(Chom,D_TC,1,1,1,1)
        if case =='Shear modulus maximization':
            h=4*Type1_h(Chom,0,1,0,1)
            D_TJ=4*Type1_DTJ(Chom,D_TC,0,1,0,1)
        if case =='minimization of modified Poisson ratio':
            h=-Type1_h(Chom,0,0,1,1)
            D_TJ=-Type1_DTJ(Chom,D_TC,0,0,1,1)     
    if type=='Type2':
        if case =='maximization of Poisson ratio':
            h=Type2_h(Chom,0,0,1,1,0,0,0,0)+Type2_h(Chom,0,0,1,1,1,1,1,1)
            D_TJ=Type2_DTJ(Chom,D_TC,0,0,1,1,0,0,0,0)+Type2_DTJ(Chom,D_TC,0,0,1,1,1,1,1,1)
        if case =='minimisation of Poisson ration under contraint on E':  #special case
            test2=inner(as_tensor(np.tensordot(np.tensordot(C_minus1(Chom),D_TC,2),C_minus1(Chom),2)),Prod_varphi(0,0,0,0))
            test3=inner(as_tensor(C_minus1(Chom)),Prod_varphi(0,0,0,0))*inner(as_tensor(C_minus1(Chom)),Prod_varphi(0,0,0,0))
            h=-Type2_h(Chom,0,0,1,1,0,0,0,0)-40/17.02*1/inner(as_tensor(C_minus1(Chom)),Prod_varphi(0,0,0,0))+20*20*1/inner(as_tensor(C_minus1(Chom)),Prod_varphi(0,0,0,0))*1/inner(as_tensor(C_minus1(Chom)),Prod_varphi(0,0,0,0))+20*20/(17.01*17.01)
            D_TJ=-Type2_DTJ(Chom,D_TC,0,0,1,1,0,0,0,0)-2*20/17.02*test2/test3+2*20*20*test2/test3*1/inner(as_tensor(C_minus1(Chom)),Prod_varphi(0,0,0,0))
    if type=='Type4':
        if case =='R0 orthotrope':
            h1=125*Type4_h(Chom,0,0,0,0,0,0,0,0)
            h2=2000*Type4_h(Chom,0,0,0,1,0,0,0,1)
            h3=500*Type4_h(Chom,0,0,1,1,0,0,1,1)
            h4=2000*Type4_h(Chom,0,0,1,1,0,1,0,1)
            h5=2000*Type4_h(Chom,0,1,0,1,0,1,0,1)
            h6=-4000*Type4_h(Chom,0,0,0,1,0,1,1,1)
            h7=2000*Type4_h(Chom,0,1,1,1,0,1,1,1)
            h8=-500*Type4_h(Chom,0,0,0,0,0,0,1,1)
            h9=-1000*Type4_h(Chom,0,0,0,0,0,1,0,1)
            h10=250*Type4_h(Chom,0,0,0,0,1,1,1,1)
            h11=-500*Type4_h(Chom,0,0,1,1,1,1,1,1)
            h12=-1000*Type4_h(Chom,0,1,0,1,1,1,1,1)
            h13=125*Type4_h(Chom,1,1,1,1,1,1,1,1)     
            hh1=Type4_h(Chom,0,0,0,0,0,0,0,0)
            hh2=4*Type4_h(Chom,0,0,0,1,0,0,0,1)
            hh3=8*Type4_h(Chom,0,0,0,1,0,1,1,1)
            hh4=4*Type4_h(Chom,0,1,1,1,0,1,1,1)
            hh5=-2*Type4_h(Chom,0,0,0,0,1,1,1,1)
            hh6=Type4_h(Chom,1,1,1,1,1,1,1,1)
            h=20*(h1+h2+h3+h4+h5+h6+h7+h8+h9+h10+h11+h12+h13)+1/(hh1+hh2+hh3+hh4+hh5+hh6)

            p1=125*Type4_DTJ(Chom,D_TC,0,0,0,0,0,0,0,0)
            p2=2000*Type4_DTJ(Chom,D_TC,0,0,0,1,0,0,0,1)
            p3=500*Type4_DTJ(Chom,D_TC,0,0,1,1,0,0,1,1)
            p4=2000*Type4_DTJ(Chom,D_TC,0,0,1,1,0,1,0,1)
            p5=2000*Type4_DTJ(Chom,D_TC,0,1,0,1,0,1,0,1)
            p6=-4000*Type4_DTJ(Chom,D_TC,0,0,0,1,0,1,1,1)
            p7=2000*Type4_DTJ(Chom,D_TC,0,1,1,1,0,1,1,1)
            p8=-500*Type4_DTJ(Chom,D_TC,0,0,0,0,0,0,1,1)
            p9=-1000*Type4_DTJ(Chom,D_TC,0,0,0,0,0,1,0,1)
            p10=250*Type4_DTJ(Chom,D_TC,0,0,0,0,1,1,1,1)
            p11=-500*Type4_DTJ(Chom,D_TC,0,0,1,1,1,1,1,1)
            p12=-1000*Type4_DTJ(Chom,D_TC,0,1,0,1,1,1,1,1)
            p13=125*Type4_DTJ(Chom,D_TC,1,1,1,1,1,1,1,1)
            m1=Type4_DTJ(Chom,D_TC,0,0,0,0,0,0,0,0)
            m2=4*Type4_DTJ(Chom,D_TC,0,0,0,1,0,0,0,1)
            m3=8*Type4_DTJ(Chom,D_TC,0,0,0,1,0,1,1,1)
            m4=4*Type4_DTJ(Chom,D_TC,0,1,1,1,0,1,1,1)
            m5=-2*Type4_DTJ(Chom,D_TC,0,0,0,0,1,1,1,1)
            m6=Type4_DTJ(Chom,D_TC,1,1,1,1,1,1,1,1)
            mm1=inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(0,0,0,0))*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(0,0,0,0))
            mm2=4*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(0,0,0,1))*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(0,0,0,1))
            mm3=8*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(0,0,0,1))*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(0,1,1,1))
            mm4=4*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(0,1,1,1))*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(0,1,1,1))
            mm5=-2*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(0,0,0,0))*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(1,1,1,1))
            mm6=inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(1,1,1,1))*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(1,1,1,1))
            D_TJ=20*(p1+p2+p3+p4+p5+p6+p7+p8+p9+p10+p11+p12+p13)+(-(m1+m2+m3+m4+m5+m6))/(mm1+mm2+mm3+mm4+mm5+mm6)**2
        if case =='R0 orthotrope_plus':
            h1=125*Type4_h(Chom,0,0,0,0,0,0,0,0)
            h2=2000*Type4_h(Chom,0,0,0,1,0,0,0,1)
            h22=-10*Type3_h(Chom,0,0,1,1)
            h3=500*Type4_h(Chom,0,0,1,1,0,0,1,1)
            h32=20*Type3_h(Chom,0,1,0,1)
            h4=2000*Type4_h(Chom,0,0,1,1,0,1,0,1)
            h5=2000*Type4_h(Chom,0,1,0,1,0,1,0,1)
            h6=-4000*Type4_h(Chom,0,0,0,1,0,1,1,1)
            h7=2000*Type4_h(Chom,0,1,1,1,0,1,1,1)
            h8=-500*Type4_h(Chom,0,0,0,0,0,0,1,1)
            h9=-1000*Type4_h(Chom,0,0,0,0,0,1,0,1)
            h10=250*Type4_h(Chom,0,0,0,0,1,1,1,1)
            h102=10*Type3_h(Chom,1,1,1,1)
            h11=-500*Type4_h(Chom,0,0,1,1,1,1,1,1)
            h12=-1000*Type4_h(Chom,0,1,0,1,1,1,1,1)
            h13=125*Type4_h(Chom,1,1,1,1,1,1,1,1)     
            hh1=Type4_h(Chom,0,0,0,0,0,0,0,0)
            hh2=4*Type4_h(Chom,0,0,0,1,0,0,0,1)
            hh3=8*Type4_h(Chom,0,0,0,1,0,1,1,1)
            hh4=4*Type4_h(Chom,0,1,1,1,0,1,1,1)
            hh5=-2*Type4_h(Chom,0,0,0,0,1,1,1,1)
            hh6=Type4_h(Chom,1,1,1,1,1,1,1,1)
            h=20*(h1+h2+h22+h3+h32+h4+h5+h6+h7+h8+h9+h10+h102+h11+h12+h13)+1/(hh1+hh2+hh3+hh4+hh5+hh6)

            p1=125*Type4_DTJ(Chom,D_TC,0,0,0,0,0,0,0,0)
            p2=2000*Type4_DTJ(Chom,D_TC,0,0,0,1,0,0,0,1)
            p22=-10*Type3_DTJ(D_TC,0,0,1,1)
            p3=500*Type4_DTJ(Chom,D_TC,0,0,1,1,0,0,1,1)
            p32=20*Type3_DTJ(D_TC,0,1,0,1)
            p4=2000*Type4_DTJ(Chom,D_TC,0,0,1,1,0,1,0,1)
            p5=2000*Type4_DTJ(Chom,D_TC,0,1,0,1,0,1,0,1)
            p6=-4000*Type4_DTJ(Chom,D_TC,0,0,0,1,0,1,1,1)
            p7=2000*Type4_DTJ(Chom,D_TC,0,1,1,1,0,1,1,1)
            p8=-500*Type4_DTJ(Chom,D_TC,0,0,0,0,0,0,1,1)
            p9=-1000*Type4_DTJ(Chom,D_TC,0,0,0,0,0,1,0,1)
            p10=250*Type4_DTJ(Chom,D_TC,0,0,0,0,1,1,1,1)
            p102=10*Type3_DTJ(D_TC,1,1,1,1)
            p11=-500*Type4_DTJ(Chom,D_TC,0,0,1,1,1,1,1,1)
            p12=-1000*Type4_DTJ(Chom,D_TC,0,1,0,1,1,1,1,1)
            p13=125*Type4_DTJ(Chom,D_TC,1,1,1,1,1,1,1,1)
            m1=Type4_DTJ(Chom,D_TC,0,0,0,0,0,0,0,0)
            m2=4*Type4_DTJ(Chom,D_TC,0,0,0,1,0,0,0,1)
            m3=8*Type4_DTJ(Chom,D_TC,0,0,0,1,0,1,1,1)
            m4=4*Type4_DTJ(Chom,D_TC,0,1,1,1,0,1,1,1)
            m5=-2*Type4_DTJ(Chom,D_TC,0,0,0,0,1,1,1,1)
            m6=Type4_DTJ(Chom,D_TC,1,1,1,1,1,1,1,1)
            mm1=inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(0,0,0,0))*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(0,0,0,0))
            mm2=4*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(0,0,0,1))*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(0,0,0,1))
            mm3=8*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(0,0,0,1))*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(0,1,1,1))
            mm4=4*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(0,1,1,1))*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(0,1,1,1))
            mm5=-2*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(0,0,0,0))*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(1,1,1,1))
            mm6=inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(1,1,1,1))*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(1,1,1,1))
            D_TJ=20*(p1+p2+p22+p3+p32+p4+p5+p6+p7+p8+p9+p10+p102+p11+p12+p13)+(-(m1+m2+m3+m4+m5+m6))/(mm1+mm2+mm3+mm4+mm5+mm6)**2
        if case=='Cauchy elasticity':
            h1=Type4_h(Chom,0,0,1,1,0,0,1,1)
            h2=-2*Type4_h(Chom,0,0,1,1,0,1,0,1)
            h3=Type4_h(Chom,0,1,0,1,0,1,0,1)

            h4=Type4_h(Chom,0,0,0,0,0,0,0,0)
            h5=-4*Type4_h(Chom,0,0,0,0,0,0,1,1)
            h6=4*Type4_h(Chom,0,0,1,1,0,0,1,1)
            h7=8*Type4_h(Chom,0,0,0,0,0,1,0,1)
            h8=-16*Type4_h(Chom,0,0,1,1,0,1,0,1)
            h9=16*Type4_h(Chom,0,1,0,1,0,1,0,1)
            h10=2*Type4_h(Chom,0,0,0,0,1,1,1,1)
            h11=-4*Type4_h(Chom,0,0,1,1,1,1,1,1)
            h12=8*Type4_h(Chom,0,1,0,1,1,1,1,1)
            h13=Type4_h(Chom,1,1,1,1,1,1,1,1)

            # h14=Type4_h(Chom,0,0,0,0,0,0,0,0)
            # h15=4*Type4_h(Chom,0,0,0,0,0,0,1,1)
            # h16=4*Type4_h(Chom,0,0,1,1,0,0,1,1)
            # h17=2*Type4_h(Chom,0,0,0,0,1,1,1,1)
            # h18=4*Type4_h(Chom,0,0,1,1,1,1,1,1)
            # h19=Type4_h(Chom,1,1,1,1,1,1,1,1)

            h=5000*(h1+h2+h3)+1/(h4+h5+h6+h7+h8+h9+h10+h11+h12+h13)

            p1=Type4_DTJ(Chom,D_TC,0,0,1,1,0,0,1,1)
            p2=-2*Type4_DTJ(Chom,D_TC,0,0,1,1,0,1,0,1)
            p3=Type4_DTJ(Chom,D_TC,0,1,0,1,0,1,0,1)

            p4=Type4_DTJ(Chom,D_TC,0,0,0,0,0,0,0,0)
            p5=-4*Type4_DTJ(Chom,D_TC,0,0,0,0,0,0,1,1)
            p6=4*Type4_DTJ(Chom,D_TC,0,0,1,1,0,0,1,1)
            p7=8*Type4_DTJ(Chom,D_TC,0,0,0,0,0,1,0,1)
            p8=-16*Type4_DTJ(Chom,D_TC,0,0,1,1,0,1,0,1)
            p9=16*Type4_DTJ(Chom,D_TC,0,1,0,1,0,1,0,1)
            p10=2*Type4_DTJ(Chom,D_TC,0,0,0,0,1,1,1,1)
            p11=-4*Type4_DTJ(Chom,D_TC,0,0,1,1,1,1,1,1)
            p12=8*Type4_DTJ(Chom,D_TC,0,1,0,1,1,1,1,1)
            p13=Type4_DTJ(Chom,D_TC,1,1,1,1,1,1,1,1)

            m4=inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(0,0,0,0))*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(0,0,0,0))
            m5=-4*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(0,0,0,0))*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(0,0,1,1))
            m6=4*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(0,0,1,1))*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(0,0,1,1))
            m7=8*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(0,0,0,0))*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(0,1,0,1))
            m8=-16*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(0,0,1,1))*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(0,1,0,1))
            m9=16*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(0,1,0,1))*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(0,1,0,1))
            m10=2*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(0,0,0,0))*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(1,1,1,1))
            m11=-4*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(0,0,1,1))*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(1,1,1,1))
            m12=8*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(0,1,0,1))*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(1,1,1,1))
            m13=inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(1,1,1,1))*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(1,1,1,1))

            # p14=Type4_DTJ(Chom,D_TC,0,0,0,0,0,0,0,0)
            # p15=4*Type4_DTJ(Chom,D_TC,0,0,0,0,0,0,1,1)
            # p16=4*Type4_DTJ(Chom,D_TC,0,0,1,1,0,0,1,1)
            # p17=2*Type4_DTJ(Chom,D_TC,0,0,0,0,1,1,1,1)
            # p18=4*Type4_DTJ(Chom,D_TC,0,0,1,1,1,1,1,1)
            # p19=Type4_DTJ(Chom,D_TC,1,1,1,1,1,1,1,1)

            # m14=inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(0,0,0,0))*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(0,0,0,0))
            # m15=4*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(0,0,0,0))*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(0,0,1,1))
            # m16=4*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(0,0,1,1))*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(0,0,1,1))
            # m17=2*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(0,0,0,0))*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(1,1,1,1))
            # m18=4*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(0,0,1,1))*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(1,1,1,1))
            # m19=inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(1,1,1,1))*inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(1,1,1,1))


            D_TJ=50000*(p1+p2+p3)-(1*(p4+p5+p6+p7+p8+p9+p10+p11+p12+p13)/(m4+m5+m6+m7+m8+m9+m10+m11+m12+m13)**2)   

    return h,D_TJ

#Initial geometry model parameters
ax=ay = 1.      # unit cell width
vol = ax*ay     # unit cell volume




#Material parameters
E = 1.          #Young's modulus
nu = 0.3        #coef poisson
    

#Initialisation of optimisation parameters 
ItMax,It=[1000,0]      #stopping criteria  
tol = 1E-14  #tolerance 
J=np.zeros(ItMax)                  #initialization of cost function
ls,ls_max=[0,10]


#Initialisation of plot values
th_plot=[]
Vol_plot=[]
J_plot=[]







