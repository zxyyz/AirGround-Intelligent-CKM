"""Actual UCA steering correlation; no ULA approximation or resolution claim."""
from common import *
from scipy.optimize import brentq
from ckm_sim.sounding import array_response

def correlation(phi,theta,ref_phi,ref_theta,c,receive=True):
    p,t=np.broadcast_arrays(phi,theta);a=np.stack([p,t],-1)
    ref=array_response(np.array([ref_phi,ref_theta]),8,c,receive)
    response=array_response(a,8,c,receive)
    return abs(np.sum(ref.conj()*response,axis=-1))**2

def first_crossing(function,maximum):
    x=np.linspace(0,maximum,18001);y=function(x)-.5;indices=np.flatnonzero((y[:-1]>0)&(y[1:]<=0))
    if not len(indices):return None
    i=indices[0];return float(brentq(lambda a:float(function(a)-.5),x[i],x[i+1],xtol=1e-14))

def calculate():
    c=cfg();wave=c['propagation']['speed_of_light_mps']/c['propagation']['carrier_hz'];radius=wave/(4*np.sin(np.pi/8));theta=np.pi/2
    h=first_crossing(lambda d:correlation(d,theta,0,theta,c),np.pi)
    ht=first_crossing(lambda d:correlation(d,theta,0,theta,c,False),np.pi)
    zen=first_crossing(lambda d:correlation(0,theta-d,0,theta,c),np.pi/2)
    assert h is not None and zen is not None and abs(ht-h)<1e-13
    angular=[]
    for deg in [0,5,10,30,45,60,75,90]:
        t=np.deg2rad(deg);delta=first_crossing(lambda d:correlation(d,t,0,t,c),np.pi)
        left=first_crossing(lambda d:correlation(0,t-d,0,t,c),t) if t else None
        right=first_crossing(lambda d:correlation(0,t+d,0,t,c),np.pi-t) if t<np.pi else None
        angular.append(dict(reference_zenith_deg=deg,azimuth_half_width_deg=float(np.rad2deg(delta)) if delta else None,azimuth_full_width_deg=float(2*np.rad2deg(delta)) if delta else None,azimuth_search_positive_deg=[0,180],toward_zenith_half_width_deg=float(np.rad2deg(left)) if left else None,toward_nadir_half_width_deg=float(np.rad2deg(right)) if right else None))
    n=c['sounding']['n_frequencies'];df=c['sounding']['delta_f_hz'];q=c['estimator']['frequency_window_q']
    return dict(carrier_hz=c['propagation']['carrier_hz'],speed_of_light_mps=c['propagation']['speed_of_light_mps'],wavelength_m=wave,n_elements=8,radius_m=radius,radius_over_wavelength=radius/wave,reference_azimuth_deg=0,reference_zenith_deg=90,half_power_level=.5,azimuth_half_power_half_width_deg=float(np.rad2deg(h)),azimuth_HPBW_full_width_deg=float(2*np.rad2deg(h)),tx_half_width_deg=float(np.rad2deg(ht)),zenith_toward_pole_half_width_deg=float(np.rad2deg(zen)),nominal_bandwidth_hz=n*df,endpoint_tone_span_hz=(n-1)*df,nominal_delay_scale_ns=1e9/(n*df),Q=q,Q_nominal_aperture_hz=q*df,Q_endpoint_tone_span_hz=(q-1)*df,Q_delay_scale_ns=1e9/(q*df),direction_dependence=angular,notes=['HPBW is reference azimuth full mainlobe width at phi=0, theta=90 degrees, not an isotropic 3D resolution.','A single-path steering-correlation width is not a MUSIC RMSE, CRLB, hard two-path limit, or superresolution guarantee.','Near zenith azimuth is unidentifiable; no half-power root within the scan is reported as null.','Horizontal UCA has exact upper/lower mirror ambiguity.','Gain scale 5 dB is retained engineering normalization, not derivable from aperture alone.','Nf*delta_f=64 MHz is the nominal definition; the actual endpoint-tone separation is 63.5 MHz.','Physical scale groups are mixed physical references and engineering gain normalization, not IEEE/3GPP mandated values.'])

def run():
    assert load(EXP/'results/baseline_reproduction.json')['passed'];start=time.perf_counter();value=calculate();save(EXP/'results/physical_resolution.json',dict(**value,wall_s=time.perf_counter()-start,epoch=time.time()))
    grid=copy.deepcopy(load(EXP/'configs/protocol.json')['scales'])
    for row in grid:
        if row['angle_deg']=='derived_full_HPBW':row['angle_deg']=value['azimuth_HPBW_full_width_deg']
        elif row['angle_deg']=='derived_half_power_half_width':row['angle_deg']=value['azimuth_half_power_half_width_deg']
        if row['id'] in ['S_phys','S_phys_half']:row['delay_ns']=value['nominal_delay_scale_ns']
        if row['id']=='S_phys_Q':row['delay_ns']=value['Q_delay_scale_ns']
    save(EXP/'configs/scales_resolved.json',dict(scales=grid,physical_resolution_sha256=sha(EXP/'results/physical_resolution.json'),protocol_sha256=sha(EXP/'configs/protocol.json'),resolution='deterministic physical definitions frozen before any new-scale scoring'))
    plot(value);stage('B_complete_C_pending',B='passed');print('PHYSICAL_SCALES',value['azimuth_HPBW_full_width_deg'],value['azimuth_half_power_half_width_deg'],value['zenith_toward_pole_half_width_deg'],flush=True)

def plot(value):
    from plot_style import setup,savefig
    plt=setup();c=cfg();fig,axes=plt.subplots(1,3,figsize=(12,3.45));d=np.linspace(0,90,1801)
    for deg in [30,45,60,75,90]:
        t=np.deg2rad(deg);axes[0].plot(d,correlation(np.deg2rad(d),t,0,t,c),label=f'{deg}°')
    axes[0].axhline(.5,color='black',lw=.7,ls='--');axes[0].axvline(value['azimuth_half_power_half_width_deg'],color='black',lw=.7,ls=':');axes[0].set(xlabel='Azimuth displacement (deg)',ylabel='Squared steering correlation',title='Azimuth cuts at reference zeniths',xlim=(0,90),ylim=(0,1.03));axes[0].legend(title='Zenith',fontsize=9)
    axes[1].plot(d,correlation(0,np.pi/2-np.deg2rad(d),0,np.pi/2,c));axes[1].axhline(.5,color='black',lw=.7,ls='--');axes[1].axvline(value['zenith_toward_pole_half_width_deg'],color='black',lw=.7,ls=':');axes[1].set(xlabel='Zenith displacement toward pole (deg)',title='Zenith cut from horizontal incidence',xlim=(0,90),ylim=(0,1.03))
    rows=[r for r in value['direction_dependence'] if r['azimuth_full_width_deg'] is not None];axes[2].plot([r['reference_zenith_deg'] for r in rows],[r['azimuth_full_width_deg'] for r in rows],'o-');axes[2].set(xlabel='Reference zenith (deg)',ylabel='Azimuth HPBW full width (deg)',title='Direction-dependent azimuth width');axes[2].text(.04,.94,'No crossing at zenith 0°, 5°',transform=axes[2].transAxes,fontsize=9,va='top')
    for a in axes:a.grid(alpha=.18)
    fig.tight_layout();savefig(fig,'array_response_resolution')

if __name__=='__main__':
    try:run()
    except Exception as e:failure('physical_resolution',e);raise
