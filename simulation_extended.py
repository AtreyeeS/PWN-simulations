import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from gammapy.irf import load_cta_irfs
from gammapy.maps import WcsGeom, MapAxis
from gammapy.modeling.models import (
    PowerLaw2SpectralModel,
    GaussianSpatialModel,
    SkyModel,
    Models,
    FoVBackgroundModel,
)
from gammapy.makers import MapDatasetMaker, SafeMaskMaker
from gammapy.modeling import Fit
from gammapy.data import Observation, DataStore
from gammapy.datasets import MapDataset, Datasets
from gammapy.utils.random import get_random_state

class Simulation():
    
    def __init__(
        self,
        livetime =1.0*u.hr,
        wobble_offset = 0.7*u.deg,
        skydir = SkyCoord(0, 0, unit="deg", frame="galactic"),
        mean_bkg = 1.0,
        sigma_bkg = 0.2,
    ):
        
        
        self.livetime = livetime
        self.wobble_offset = wobble_offset
        self.skydir = skydir # The center of the source
        self.mean_bkg = mean_bkg
        self.sigma_bkg = sigma_bkg
        self.bkg_norms = []
        
    
    def get_irf(self):
        data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1")
        obs = data_store.obs(47828)
        irfs = {'aeff': obs.aeff,
            'edisp': obs.edisp,
            'psf': obs.psf,
            'bkg': obs.bkg}
        return irfs
    
    def create_empty(self, name):
        energy_reco = MapAxis.from_energy_bounds(energy_min=300.0*u.GeV, 
                                                 energy_max=10.0*u.TeV, nbin=10,
                                                name='energy')
        geom = WcsGeom.create(skydir=self.skydir,
                              binsz=0.05,
                              width=(8, 8),
                              frame="galactic",
                              axes=[energy_reco],
                              )

        empty = MapDataset.create(geom, name=name)
        return empty
    
    
    def sky_model(self, sigma=0.3, A0=1e-12):
        spatial_model = GaussianSpatialModel(frame="galactic")
        spatial_model.lon_0.value = self.skydir.galactic.l.value
        spatial_model.lat_0.value = self.skydir.galactic.b.value
        spatial_model.sigma.value = sigma
        spectral_model = PowerLaw2SpectralModel()
        amplitude = A0 * sigma * sigma
        spectral_model.amplitude.value = amplitude
        model_simu = SkyModel(spatial_model=spatial_model,
                              spectral_model=spectral_model,
                                name="model-simu",
                              )
        
        return model_simu
        

    def simulate_single(self, pointing, models=None, empty=None, random_state="random-seed"):
        obs = Observation.create(pointing=pointing, 
                                 livetime=self.livetime, 
                                 irfs=self.get_irf())
        maker = MapDatasetMaker(selection=["exposure", "background", "psf", "edisp"])
        maker_safe_mask = SafeMaskMaker(methods=["offset-max"], offset_max=2.0 * u.deg)
        
        
        dataset = maker.run(empty, obs)
        dataset = maker_safe_mask.run(dataset, obs)
        bkg_model = FoVBackgroundModel(dataset_name=dataset.name)
        
        dataset.models = Models([bkg_model,models])
        
        
        #fluctuate the background - gaussian fluctuations
        bkg_factor = np.random.normal(self.mean_bkg, self.sigma_bkg)
        while bkg_factor<0.0:
            bkg_factor = np.random.normal(self.mean_bkg, self.sigma_bkg)
        dataset.models[0].spectral_model.norm.value = bkg_factor
        self.bkg_norms.append(bkg_factor)
        
        #TODO: Discuss correct behaviour
        
        #Poission fluctuate only the source
        """
        random_state = get_random_state(random_state)
        npred = dataset.npred_signal() 
        npred.data = random_state.poisson(npred.data)
        dataset.counts = npred + dataset.background
        """
        
        
        #Poission fluctuate source + background
        dataset.fake()
        
        
        dataset.models = Models([bkg_model]) #remove the model on the dataset
        
        return dataset
    
    
    def run(self, n_obs=10, sigma=0.3*u.deg, A0=5e-13):
        pos1 = SkyCoord(self.skydir.galactic.l + self.wobble_offset, self.skydir.galactic.b, 
                        frame="galactic")
        pos2 = SkyCoord(self.skydir.galactic.l - self.wobble_offset, self.skydir.galactic.b, 
                        frame="galactic")
        pos3 = SkyCoord(self.skydir.galactic.l, self.skydir.galactic.b + self.wobble_offset, 
                        frame="galactic")
        pos4 = SkyCoord(self.skydir.galactic.l, self.skydir.galactic.b - self.wobble_offset, 
                        frame="galactic")
        
        datasets = Datasets()
        
        for j,apos in enumerate([pos1, pos2, pos3, pos4]):
            print("Pointing position: \n", apos)
            for i in range(n_obs):
                empty = self.create_empty(name=f"dataset-{j}-{i}")
                models = self.sky_model(sigma.value, A0)
                dataset = self.simulate_single(pointing=apos, 
                                               models=models, 
                                               empty=empty)
                datasets.append(dataset)
                
        return datasets
        