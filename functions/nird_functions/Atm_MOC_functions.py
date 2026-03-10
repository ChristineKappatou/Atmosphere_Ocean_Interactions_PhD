# Functions ---------------------------------------------

### Noresm needs a special function to remove the antartcica feature

def dpres_zm_amon_noresm(experiment, ds, i,  ps_data, pcoord="plev"):
    """
    Compute layer pressure thickness dp for pressure-level data, 
    removing extrapolated levels above surface pressure.

    Returns dp with dimensions (time, lev-1, lat, lon).
    """

    p = ds[pcoord]

    # Ensure levels are increasing downward (low p → high p)
    if not (p.diff(pcoord) > 0).all():
        ds = ds.sortby(pcoord)
        p = ds[pcoord]

    # Compute dp between midpoints
    dp_vals = p.diff(pcoord)  # length nlev-1

    # Align dp with the upper levels (0..nlev-2)
    dp_vals = dp_vals.assign_coords(
        {pcoord: p.isel({pcoord: slice(0, -1)})}
    )

    # Broadcast dp to (time, lev-1, lat, lon)
    dp = dp_vals.broadcast_like(ds["va"].isel({pcoord: slice(0, -1)}))

    # ---------------------------
    # REMOVE EXTRAPOLATED VALUES
    # ---------------------------
    # Surface pressure

    ps = ps_data[i]
    # ps = ds[ps_name]

    # Expand plev to same dims
    p_4d = p.broadcast_like(ds["va"])
    p_4d = p_4d.isel({pcoord: slice(0, -1)})  # match dp dims

    # Mask where p > ps (extrapolated)
    mask = p_4d > ps
    dp = dp.where(~mask)

    return dp

def calc_atm_MOC_amon_noresm(experiment, ds, i, ps_data, var="va"):
    """
    Calculate cumulative vertical integral of meridional velocity (MOC streamfunction),
    removing pressure levels above surface pressure.
    """

    Earth_rad = 6.371e6  # meters
    g = 9.81
    lat_rad = np.deg2rad(ds.lat)
    coslat = np.cos(lat_rad)

    va = ds[var].squeeze()

    # Sort pressure levels (low → high)
    if not (ds.plev.diff("plev") > 0).all():
        ds = ds.sortby("plev")
        va = va.sortby("plev")

    # Compute dp with level filtering
    dp = dpres_zm_amon_noresm(experiment, ds, i, ps_data)

    # Mask velocity the same way (remove extrapolated values)
    ps = ps_data[i]

    plev_4d = ds.plev.broadcast_like(va)
    va_masked = va.where(plev_4d <= ps)

    # Multiply v * dp only for valid levels
    vdp = va_masked.isel(plev=slice(0, -1)) * dp

    # Expand cos(lat)
    coslat = coslat.expand_dims({'plev': vdp.plev}).transpose('plev', 'lat')

    # Cumulative integral downward
    cumint = vdp.cumsum(dim="plev")

    # Convert to Sv
    atm_moc = (
        1e-10 * 2 * np.pi * Earth_rad / g
        * coslat * cumint
    )

    return atm_moc, va_masked

# Functions for the Amon data (what we will use)
def dpres_zm_amon(ds, pcoord="plev"):
    """
    Compute layer pressure thickness dp for pressure-level data.
    Returns dp with dimensions (year, lev-1, lat, lon).

    Assumes pressure levels are midpoints of layers.
    Uses finite-difference between adjacent levels:
         dp[i] = p[i+1] - p[i]
    with sign automatically handled by sorting levels if necessary.
    """

    p = ds[pcoord]

    # Ensure p is increasing downward (top -> bottom)
    # Typical climate convention: top has LOW p, bottom has HIGH p.
    if not (p.diff("plev") > 0).all():
        # reverse if needed
        ds = ds.sortby(pcoord)
        p = ds[pcoord]

    # Compute dp between midpoints
    dp_vals = p.diff("plev")    # length nlev-1

    # Give dp same lev coordinate as upper edges (0..nlev-2)
    dp_vals = dp_vals.assign_coords(
        {pcoord: ds[pcoord].isel(plev=slice(0, -1))}
    )

    # Expand to (year, lev-1, lat, lon)
    dp = dp_vals.broadcast_like(ds["va"].isel(plev=slice(0, -1)))

    return dp

def calc_atm_MOC_amon(_ds, var="va"):
    """
    Calculate cumulative vertical integral of meridional velocity 
    (MOC streamfunction) from top to bottom.
    """
    Earth_rad = 6.371e6  # meters
    g = 9.81 # m/s2
    lat_rad = np.deg2rad(_ds.lat)
    coslat = np.cos(lat_rad)

    va = _ds[var].squeeze()  # (time, lev, lat, lon or lat)

    # Ensure pressure is ascending (top -> bottom)
    if not (_ds.plev.diff("plev") > 0).all():
        _ds = _ds.sortby("plev")
        va = va.sortby("plev")

    dp = dpres_zm_amon(_ds)  # (time, lev, lat, lon)
    
    # Multiply by delta p (Pa)
    vdp = va.isel(plev=slice(0, -1)) * dp
    coslat = coslat.expand_dims({'plev': vdp.plev}).transpose('plev', 'lat')
    
    # Cumulative sum along lev dimension from top of atmosphere downward
    # Assuming lev is ordered from top (low p) to bottom (high p)
    cumint = vdp.cumsum(dim="plev")  
    # Apply scaling: convert Pa*m/s -> Sv (1e-15 m³/s)
    result = 1e-10 * 2 * np.pi * Earth_rad /g * coslat*cumint

    return result