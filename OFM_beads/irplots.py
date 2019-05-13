from matplotlib import pyplot as plt
import numpy as np
import scipy
import scipy.integrate
import scipy.interpolate as ip
import multiprocessing as mp
from centres_plots import intensitiesRadius

def nikon():
    cal=np.load('nikon_beads.npy')
    plt.figure()
    im=np.repeat(cal[:,:,1],5, axis=0)
    im=np.repeat(im, 20, axis=1)
    plt.imshow(im, cmap='gray')
    plt.show()

def ofm():
    cal=np.load('ofm_beads_cal.npy')
    plt.figure()
    im=np.repeat(cal[:,:,1],5, axis=0)
    im=np.repeat(im, 20, axis=1)
    plt.imshow(im, cmap='gray')
    plt.show()

def csf(r,z,maxk):
    """
    1pix = 42nm on ofm
    1pix = ???? on nikon
    """
    #k=2*np.pi/(500e-9)
    expf=-1j*z/2*500e-9
    fx=lambda x: scipy.special.j0(2*x*r)*np.exp(-expf*x**2)*x
    return scipy.integrate.quad(lambda x: np.real(fx(x)),
                                    0, maxk, limit=400)+1j*np.array(scipy.integrate.quad(
                                            lambda x: np.imag(fx(x)),
                                    0, maxk, limit=400))
def beadFn0(r, R=1e-6/2.0):
    """
    Transmission through bead centered at 0,0 with diameter 1um
    """
    return np.heaviside(R-r,0)*(np.exp(-4.5*(r/R)**2)-1)+1
    #if r<=R:
     #   return 0
        #return 
    #
    #    return np.exp(-4.5*(r/R)**2)
   #     return 0
    return 1.0

def psf(r,z, maxk):
    #at large r psf->0 (also some of the approximations in deriving may be invalid)
    #if(r>2e-3):
    #    return 0
    return np.abs(csf(r,z,maxk)[0])**2*(500e-9)**2

def convolvedIr(r, z, maxk, beadr=1e-6/2.0, maxR=1e-3, psfdata=None):
    if psfdata is None:
        def eint(R, eps):
            return psf(np.sqrt(R**2+r**2-2*r*R*np.cos(eps)), z, maxk)*R*beadFn0(R)
    else:
        psff=ip.interp1d(psfdata[:,0],psfdata[:,1])
        def eint(R, eps):
            return psff(np.sqrt(R**2+r**2-2*r*R*np.cos(eps)))*R*beadFn0(R)
    return scipy.integrate.nquad(eint, [[0, maxR], [0, 2*np.pi]])[0]
#convolvedIr(1e-5, 0, 1e7)    

def tableRowHelper(r,z,dkp, psfdata=None):
    return [z,r,convolvedIr(r,z,dkp,psfdata=psfdata)]

def precomputePSF(z, maxk, maxR=1e-5, steps=1500):
    data=[]
    for r in np.linspace(0,maxR, steps):
        data.append([r,psf(r,z,maxk)])
    data.append([maxR+1e-6, 0])
    data.append([1,0])
    return np.array(data)
    
def exactRow(z, maxk, rsteps, rmax=3e-6):
    psfd=precomputePSF(z,maxk)
    out=[]
    for r in np.linspace(0,rmax, rsteps):
        out.append(convolvedIr(r,z,maxk, psfdata=psfd))
    return out

def objectImage(maxs=2e-5, steps=2000):
    image=[]
    #would be much faster if this was created as circles (sqrt faster)
    #but it is fast enough so whatever
    for i in np.linspace(0,maxs,steps):
        row=[]
        image.append(row)
        for j in np.linspace(0,maxs,steps):
            row.append(beadFn0(np.sqrt(((maxs/2-i)**2+(maxs/2-j)**2))))
    return np.array(image)

def otf(k,z, maxk, f=1e-3):
    #if k > kmax:
     #   return 0
    kappa=2*np.pi/500e-9
    #w=z
    alpha=np.arctan(1)
    w=-f-z*np.cos(alpha)+np.sqrt(f**2+2*f*z+z**2*np.cos(alpha)**2)
    #w=z
    print(f"(w-z)/z = {(w-z)/z}")
    ar=4*np.pi*w*maxk*k/kappa*(1-k/maxk)
    zp=np.heaviside(np.abs(ar), 0)*(scipy.special.j1(ar)/ar-0.5)+0.5
    otf0=2/np.pi*(np.arccos(k/maxk)-k/maxk*np.sqrt(1.0-(k/maxk)**2))
    if w == 0:
        return otf0
    return np.nan_to_num(zp*otf0)


def resrow(z, maxk, ks, fto):
    otfd=otf(ks, z, maxk)
    ftim=otfd#*fto
    im=np.abs(np.fft.ifft2(ftim))
    #plt.imshow(im)
    #def intensitiesRadius(image, x0, y0, binsize, maxr=-1 ,raw=False, scale=1.0)
    return intensitiesRadius(im,0, 0,1,maxr=len(im))[:,1]#len(im)/2.0,len(im)/2.0,1, raw=False)[:,1]

def exactFT():
    zrange=1e-6
    #data=[]
    pool=mp.Pool(7)
    tasks=[]
    zsteps=50
    rsteps=30
    maxk=2*np.pi/500e-9
    maxr=2e-5
    imsize=800
    imsizer=1e-6
    #first get the FT of the bead
    #takes ~12s
    #should really use Hankel to make this massively faster...
    im=objectImage(imsizer,imsize)
    ftim=np.fft.fft2(im)
    ftf=np.fft.fftfreq(imsize, d=imsizer/imsize)
    ks=[]
    #there is a numpy way to do this...
    for i in range(0,imsize):
        row=[]
        ks.append(row)
        for j in range(0,imsize):
            row.append(np.sqrt(ftf[j]**2+ftf[i]**2))
    ks=np.array(ks)
    print("Ks loaded")
    res=[]
    tasks=[]

    for z in np.linspace(-zrange,zrange,zsteps):
        tasks.append((z,maxk, ks,ftim))
    res=pool.starmap(resrow,tasks)
    pool.close()
    pool.join()
    return np.array(res)

#q=exactFT()

def exact():
    zrange=2e-6
    #data=[]
    pool=mp.Pool(7)
    tasks=[]
    zsteps=7
    rsteps=30
    maxk=2*np.pi/500e-9
    maxr=1e-6
    for z in np.linspace(-zrange,zrange,zsteps):
        #exactRow(z,maxk,rsteps,maxr)
        tasks.append((z, maxk, rsteps, maxr))
        #row=[]
        #psfd=precomputePSF(z, maxk)
        #for r in np.linspace(0, 3e-6, rsteps):
        #    tasks.append((r,z,1e7,psfd))
            #row.append([r,convolvedIr(r,z,1e7)])
        #    print("Queued")
        #data.append(row)
    rawd=pool.starmap(exactRow,tasks)
    pool.close()
    pool.join()
    return np.array(rawd)#, np.array(psfs)

#qe=exact()