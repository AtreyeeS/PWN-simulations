from gammapy.makers import RingBackgroundMaker
from gammapy.estimators import ExcessMapEstimator
from gammapy.maps import Map
from gammapy.datasets import MapDatasetOnOff
import astropy.units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import numpy as np
from regions import CircleSkyRegion
from scipy.stats import norm

def extract_ring(datasets,
                 source_pos,
                 radius=0.3*u.deg,
                 r_in="0.5 deg",
                 width="0.3 deg"):
    geom = datasets[0].counts.geom
    energy_axis = datasets[0].counts.geom.axes["energy"]
    geom_image = geom.to_image().to_cube([energy_axis.squash()])

    # Make the exclusion mask
    regions = CircleSkyRegion(center=source_pos, radius=radius)
    exclusion_mask = Map.from_geom(geom_image)
    exclusion_mask.data = geom_image.region_mask([regions], inside=False)

    ring_maker = RingBackgroundMaker(
        r_in=r_in, width=width, exclusion_mask=exclusion_mask
    )

    energy_axis_true = datasets[0].exposure.geom.axes["energy_true"]
    stacked_on_off = MapDatasetOnOff.create(
        geom=geom_image, energy_axis_true=energy_axis_true, name="stacked"
    )

    for dataset in datasets:
        dataset_on_off = ring_maker.run(dataset.to_image())
        stacked_on_off.stack(dataset_on_off)

    return stacked_on_off




