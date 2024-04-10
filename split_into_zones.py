import numpy as np

from alpaos_channel_generator.alpaos_fortran import *

from np2Geotiff import arr2Geotiff

def split(alp, zones):
    """
    Function to split an alpoas into zones to allow parallel computing
    ! Make sure that zones are always divided by a channel (H = 0) !
    """
    
    zone_ids                    = [z for z in np.unique(zones) if z > 0]
    alps                        = []

    for zone_id in zone_ids:

        # get the zone and its boundaries
        ij                      = np.column_stack(np.where(zones == zone_id))
        left                    = np.min(ij[:,1])
        right                   = np.max(ij[:,1])+1
        bottom                  = np.max(ij[:,0])+1
        up                      = np.min(ij[:,0])

        # ensure that it the rows and cols can be divided by 8
        rows                    = bottom - up
        cols                    = right - left
        rows_to_add             = 8 - rows % 8
        cols_to_add             = 8 - cols % 8
        bottom                  += rows_to_add
        right                   += cols_to_add

        # ensure a border around the domain
        up                      -= 4
        left                    -= 4
        bottom                  += 4
        right                   += 4

        if up < 0:
            bottom += abs(up)
        if left < 0:
            right += abs(left)
        if right > alp.H.shape[1]:
            left -= right - alp.H.shape[1]
        if bottom > alp.H.shape[0]:
            up -= bottom - alp.H.shape[0]

        # split the domain and only keep the zone
        zones_zone              = zones[up:bottom, left:right]
        domain_zone             = alp.domain[up:bottom, left:right].copy()
        domain_zone[:4,:]       = 0
        domain_zone[-4:,:]      = 0
        domain_zone[:,:4]       = 0
        domain_zone[:,-4:]      = 0
        zone_mask               = (zones_zone > 0) & (zones_zone != zone_id)
        domain_zone[zone_mask]  = 0

        # split the other arrays
        chan_zone               = alp.chan[up:bottom, left:right].copy()
        H_zone                  = alp.H[up:bottom, left:right]*domain_zone.copy()
        K_zone                  = alp.K[up:bottom, left:right]*domain_zone.copy()
        pl_zone                 = alp.pl[up:bottom, left:right]*domain_zone.copy()
        Z_zone                  = alp.Z[up:bottom, left:right]*domain_zone.copy()
        H0_zone                 = alp.H0[up:bottom, left:right]*domain_zone.copy()

        bounds                  = [up, bottom, left, right]

        # save tmp tif for pyshed inititation
        tif_fn_zone              = f'Churute_tmp_{zone_id}.tif'
        arr2Geotiff(domain_zone, tif_fn_zone, [0, domain_zone.shape[0]], 1, 32717)

        # create the alpaos object
        alp_zone                = AlpaosChannelCreator(H_zone, K_zone, chan_zone, pl_zone, domain_zone,
                                                       Z=Z_zone, meta = alp.meta, H0 = H0_zone,
                                                       resolution = alp.resolution, fn_tif = tif_fn_zone)

        alps.append([alp_zone, bounds])


    return alps

def merge(alp_zones, alp):
    """
    Function to merge alpaos objects back into one
    :param alps:
    :param zones:
    :return:
    """

    # get the zone ids
    zone_ids                    = range(1, len(alp_zones)+1)

    H                           = np.zeros_like(alp.H)
    Z                           = np.zeros_like(alp.Z)
    tau                         = np.zeros_like(alp.H)
    pl                          = np.zeros_like(alp.H)
    newchan                     = np.zeros_like(alp.H)
    chan                        = np.zeros_like(alp.H)

    for zone_id in zone_ids:
        Hnew                    = np.zeros_like(alp.H)
        Znew                    = np.zeros_like(alp.H)
        taunew                  = np.zeros_like(alp.H)
        plnew                   = np.zeros_like(alp.H)
        newchannew              = np.zeros_like(alp.H)
        channew                 = np.zeros_like(alp.H)

        bounds                  = alp_zones[zone_id-1][1]

        Hnew[bounds[0]:bounds[1], bounds[2]:bounds[3]]          = alp_zones[zone_id-1][0].H
        Znew[bounds[0]:bounds[1], bounds[2]:bounds[3]]          = alp_zones[zone_id-1][0].Z
        taunew[bounds[0]:bounds[1], bounds[2]:bounds[3]]        = alp_zones[zone_id-1][0].tau
        plnew[bounds[0]:bounds[1], bounds[2]:bounds[3]]         = alp_zones[zone_id-1][0].pl
        newchannew[bounds[0]:bounds[1], bounds[2]:bounds[3]]    = alp_zones[zone_id-1][0].newchan
        channew[bounds[0]:bounds[1], bounds[2]:bounds[3]]       = alp_zones[zone_id-1][0].chan

        H                       = np.maximum(H, Hnew)
        pl                      = np.maximum(pl, plnew)
        newchan                 = np.maximum(newchan, newchannew)
        chan                    = np.maximum(chan, channew)
        Z[Znew != 0]            = Znew[Znew != 0]
        tau[taunew != 0]        = taunew[taunew != 0]

    alp.H                       = H
    alp.Z                       = Z
    alp.tau                     = tau
    alp.tau_crit                = alp_zones[0][0].tau_crit
    alp.pl                      = pl
    alp.newchan                 = newchan
    alp.chan                    = chan

    return alp