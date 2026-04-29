### **Table of Contents**

<table>
  <thead>
    <tr>
      <th>Script Name</th>
      <th>Description</th>
      <th>Needed Modules</th>
      <th>Needed Data</th>
      <th>Data Location on GH</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1.tas_NIRD.ipynb</td>
      <td>Calculates the surface air temperature.</td>
      <td>common_NIRD_functions.py</td>
      <td>ta_data.zip</td>
      <td>data>NIRD_script_data</td>
    </tr>
    <tr>
      <td>2.ta_NIRD.ipynb</td>
      <td>Calculates the air temperature.</td>
      <td>common_NIRD_functions.py</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>3.ua_NIRD.ipynb</td>
      <td>Calculates the zonal winds</td>
      <td>common_NIRD_functions.py</td>
      <td>ua_Amon_CESM2_abrupt-4xCO2_r1i1p1f1_gn_000101-015012.nc</td>
      <td>download from <a href="https://metagrid.esgf-west.org/search">ESGF</a> </td>
    </tr>
    <tr>
      <td>4.Atm_MOC_NIRD.ipynb</td>
      <td>Calculates the Meridional Overturning Circulation in the atmosphere</td>
      <td>common_NIRD_functions.py, Atm_MOC_functions.py</td>
      <td>Atm_MOC_data.zip, atm_h_t_local.zip, va_Amon_CESM2_abrupt-4xCO2_r1i1p1f1_gn_000101-015012.nc, </td>
      <td>data>NIRD_script_data, data>local_data, download from <a href="https://metagrid.esgf-west.org/search">ESGF</a></td>
    </tr>
    <tr>
      <td>5.Winds_NIRD.ipynb</td>
      <td>Calculates the changes in the winds under x4CO2</td>
      <td>common_NIRD_functions.py</td>
      <td>wind_data.zip, ua_Amon_CESM2_abrupt-4xCO2_r1i1p1f1_gn_000101-015012.nc, va_Amon_CESM2_abrupt-4xCO2_r1i1p1f1_gn_000101-015012.nc</td>
      <td>data>NIRD_script_data, download from <a href="https://metagrid.esgf-west.org/search">ESGF</a></td>
    </tr>
    <tr>
      <td>6.Oc_MOC_NIRD.ipynb</td>
      <td>calculates the Ocean Meridional Overturning Circulation. Focus on the AMOC and the Deacon cell.</td>
      <td>common_NIRD_functions.py, ocean_functions.py</td>
      <td>Oc_MOC_data.zip, thetao_z_data</td>
      <td>data>NIRD_script_data</td>
    </tr>
    <tr>
      <td>7.vT_Ocean_NIRD.ipynb</td>
      <td>Calculates the vT contours and the OHT in the direct way</td>
      <td>ocean_functions.py, common_NIRD_functions.py</td>
      <td>v_T_data.zip (contains conservative_regridding and nearest_s2d_regridding), vo_Omon_CESM2_abrupt-4xCO2_r1i1p1f1_gn_000101-015012.nc</td>
      <td>data>NIRD_script_data, download from <a href="https://metagrid.esgf-west.org/search">ESGF</a></td>
    </tr>
    <tr>
      <td>8.OHC_NIRD.ipynb</td>
      <td>Calculates the Ocean Heat Content (OHC) at different levels</td>
      <td>ocean_functions.py</td>
      <td>OHC_data.zip</td>
      <td>data>NIRD_script_data</td>
    </tr>
    <tr>
      <td>9.tos_NIRD.ipynb</td>
      <td>Calculates the temperature on the ocean surface</td>
      <td>common_NIRD_functions.py, ocean_functions.py</td>
      <td></td>
      <td></td>
    </tr>

  </tbody>
</table>
