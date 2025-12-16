import morsaik as kdi
import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from os import makedirs
from os.path import exists
import itertools
import gc
from sys import argv

from morsaik.infer import fourmer_trajectory_from_rate_constants as infer_fourmer_trajectory_from_rate_constants

def setup_Tkachenko_scenario(
        motiflength : int,
        maximum_ligation_window_length : int,
        time_unit : kdi.Unit,
    ):
    alphabet = list(np.arange(2))#20
    motif_production_array = np.zeros((3,2,2,3)*2)
    motif_production_array[0,0,0,0,0,1,1,0] = 0.5
    motif_production_array[0,1,1,0,0,0,0,0] = 0.5
    motif_production_array[0,0,1,0,0,0,1,0] = 3
    motif_production_array[0,1,0,0,0,1,0,0] = 3
    motif_production_rate_constants = kdi._array_to_motif_production_vector(
        motif_production_array,
        motiflength,
        alphabet,
        kdi.make_unit('L')**2/kdi.make_unit('mol')**2/time_unit,
        maximum_ligation_window_length
    )

    breakage_rate_constants = kdi.MotifBreakageVector(motiflength,alphabet,1./time_unit)
    breakage_vector_dct = kdi._create_empty_motif_breakage_dct(motiflength, alphabet)
    breakage_vector_array = np.zeros((3,2,2,3))
    breakage_vector_array[0,0,0,0] += .001
    breakage_vector_array[0,1,1,0] += .001 # for 0.00 also interesting oscillations
    breakage_vector_array[0,0,1,0] += .04
    breakage_vector_array[0,1,0,0] += .01 # for 0.00 also interesting oscillations
    breakage_vector_dct = kdi._array_to_motif_breakage_vector(
        breakage_vector_array,
        motiflength,
        alphabet,
        1./time_unit
    )
    breakage_rates = breakage_rate_constants(breakage_vector_dct)

    motif_vector_array = np.zeros((3,2,3,3))
    motif_vector_array[0,0,0,0] = 0.07
    motif_vector_array[0,1,0,0] = 0.09
    motif_vector_array[0,0,1,0] = 0.001
    motif_vector_array[0,1,1,0] = 0.01
    initial_concentrations_dct = kdi._array_to_motif_vector_dct(motif_vector_array,
                                motiflength,
                                alphabet,
                                )
    initial_motif_concentrations_vector = kdi.MotifVector(motiflength,alphabet,kdi.make_unit('mol')/kdi.make_unit('L'))
    initial_motif_concentrations_vector = initial_motif_concentrations_vector(initial_concentrations_dct)
    return alphabet, motif_production_rate_constants, breakage_rates, initial_motif_concentrations_vector

def setup_monomer_dimer_system(
        motiflength : int,
        maximum_ligation_window_length : int,
        time_unit : kdi.Unit,
    ):
    alphabet = ['N']
    motif_production_array = np.zeros((3,2,2,3)*2)
    motif_production_array = np.zeros((2,1,1,2)*2)
    motif_production_array[0,0,0,0,0,0,0,0] = 1.
    motif_production_rate_constants = kdi._array_to_motif_production_vector(
        motif_production_array,
        motiflength,
        alphabet,
        kdi.make_unit('L')**2/kdi.make_unit('mol')**2/time_unit,
        maximum_ligation_window_length
    )
    complements = [0,]
    breakage_rate_constants = kdi.MotifBreakageVector(motiflength,alphabet,1./time_unit)
    breakage_vector_dct = kdi._create_empty_motif_breakage_dct(motiflength, alphabet)
    breakage_vector_array = np.zeros((3,2,2,3))
    breakage_vector_array = np.zeros((2,1,1,2))
    breakage_vector_dct = kdi._array_to_motif_breakage_vector(
        breakage_vector_array,
        motiflength,
        alphabet,
        1./time_unit
    )
    breakage_rates = breakage_rate_constants(breakage_vector_dct)

    motif_vector_array = np.zeros((3,2,3,3))
    motif_vector_array = np.zeros((2,1,2,2))
    motif_vector_array[0,0,0,0] = 1.e-2
    motif_vector_array[0,0,1,0] = 1.e-5
    initial_concentrations_dct = kdi._array_to_motif_vector_dct(motif_vector_array,
                                motiflength,
                                alphabet,
                                )
    initial_motif_concentrations_vector = kdi.MotifVector(motiflength,alphabet,'mol/L')
    initial_motif_concentrations_vector = initial_motif_concentrations_vector(initial_concentrations_dct)
    return alphabet, motif_production_rate_constants, breakage_rates, initial_motif_concentrations_vector

def setup_LotkaVolterra_system(
        motiflength : int,
        maximum_ligation_window_length : int,
        time_unit : kdi.Unit,
    ):
    alphabet = ['A','T']
    motif_production_array = np.zeros((3,2,2,3)*2)
    motif_production_array[0,0,0,0,0,1,1,0] = 3
    motif_production_array[0,1,1,0,0,0,0,0] = 3
    motif_production_rate_constants = kdi._array_to_motif_production_vector(
        motif_production_array,
        motiflength,
        alphabet,
        kdi.make_unit('L')**2/kdi.make_unit('mol')**2/time_unit,
        maximum_ligation_window_length
    )
    breakage_rate_constants = kdi.MotifBreakageVector(motiflength,alphabet,1./time_unit)
    breakage_vector_dct = kdi._create_empty_motif_breakage_dct(motiflength, alphabet)
    breakage_vector_array = np.zeros((3,2,2,3))
    breakage_vector_array[0,0,0,0] += .01
    breakage_vector_array[0,1,1,0] += .01 
    breakage_vector_dct = kdi._array_to_motif_breakage_vector(
        breakage_vector_array,
        motiflength,
        alphabet,
        1./time_unit
    )
    breakage_rates = breakage_rate_constants(breakage_vector_dct)

    motif_vector_array = np.zeros((3,2,3,3))
    motif_vector_array[0,0,0,0] = 0.08
    motif_vector_array[0,1,0,0] = 0.1
    motif_vector_array[0,0,1,0] = 0.01
    initial_concentrations_dct = kdi._array_to_motif_vector_dct(motif_vector_array,
                                motiflength,
                                alphabet,
                                )
    initial_motif_concentrations_vector = kdi.MotifVector(motiflength,alphabet,'mol/L')
    initial_motif_concentrations_vector = initial_motif_concentrations_vector(initial_concentrations_dct)
    return alphabet, motif_production_rate_constants, breakage_rates, initial_motif_concentrations_vector

def setup_spontaneous_symmetry_break_scenario(
        motiflength : int,
        maximum_ligation_window_length : int,
        time_unit : kdi.Unit,
    ):
    alphabet = ['A','T']
    motif_production_array = np.zeros((3,2,2,3)*2)
    motif_production_array[0,0,0,0,0,1,1,0] = 3
    motif_production_array[0,1,1,0,0,0,0,0] = 3
    motif_production_array[0,0,1,0,0,0,1,0] = 3
    motif_production_array[0,1,0,0,0,1,0,0] = 3
    motif_production_rate_constants = kdi._array_to_motif_production_vector(
        motif_production_array,
        motiflength,
        alphabet,
        kdi.make_unit('L')**2/kdi.make_unit('mol')**2/time_unit,
        maximum_ligation_window_length
    )
    breakage_rate_constants = kdi.MotifBreakageVector(motiflength,alphabet,1./time_unit)
    breakage_vector_dct = kdi._create_empty_motif_breakage_dct(motiflength, alphabet)
    breakage_vector_array = np.zeros((3,2,2,3))
    breakage_vector_array[0,0,0,0] += .01
    breakage_vector_array[0,1,1,0] += .01
    breakage_vector_array[0,0,1,0] += .01
    breakage_vector_array[0,1,0,0] += .01
    breakage_vector_dct = kdi._array_to_motif_breakage_vector(
        breakage_vector_array,
        motiflength,
        alphabet,
        1./time_unit
    )
    breakage_rates = breakage_rate_constants(breakage_vector_dct)
    motif_vector_array = np.zeros((3,2,3,3))
    motif_vector_array[0,0,0,0] = 0.1
    motif_vector_array[0,1,0,0] = 0.1
    motif_vector_array[0,0,1,0] = 0.005
    motif_vector_array[0,1,2,0] = 0.01
    motif_vector_array[0,1,1,0] = 0.01
    motif_vector_array[0,0,2,0] = 0.01
    initial_concentrations_dct = kdi._array_to_motif_vector_dct(motif_vector_array,
                                motiflength,
                                alphabet,
                                )
    initial_motif_concentrations_vector = kdi.MotifVector(motiflength,alphabet,'mol/L')
    initial_motif_concentrations_vector = initial_motif_concentrations_vector(initial_concentrations_dct)
    return alphabet, motif_production_rate_constants, breakage_rates, initial_motif_concentrations_vector

def setup_9999_scenario(
        motiflength : int,
        maximum_ligation_window_length : int,
        time_unit : kdi.Unit,
    ):
    alphabet = ['A','T']
    motif_production_array = np.zeros((3,2,2,3)*2)
    complements = [1,0]
    motif_production_rate_constants = kdi.get.motif_production_rate_constants_from_strand_reactor_parameters(
        '9999_99_99__99_99_99',
        motiflength,
        complements,
        maximum_ligation_window_length = maximum_ligation_window_length
    )
    breakage_rate_constants = kdi.MotifBreakageVector(motiflength,alphabet,1./time_unit)
    breakage_vector_dct = kdi._create_empty_motif_breakage_dct(motiflength, alphabet)
    breakage_vector_array = np.zeros((3,2,2,3))
    breakage_rates = kdi.get.motif_breakage_rate_constants_from_strand_reactor_parameters(
        '9999_99_99__99_99_99',
        motiflength,
        complements,
    )
    motif_vector_array = np.zeros((3,2,3,3))
    motif_vector_array[0,0,2,0] = 0.01
    initial_concentrations_dct = kdi._array_to_motif_vector_dct(motif_vector_array,
                                motiflength,
                                alphabet,
                                )
    initial_motif_concentrations_vector = kdi.MotifVector(motiflength,alphabet,'mol/L')
    initial_motif_concentrations_vector = initial_motif_concentrations_vector(initial_concentrations_dct)
    return alphabet, motif_production_rate_constants, breakage_rates, initial_motif_concentrations_vector

def setup_zebra_i_scenario(
        param_file_no : int,
        motiflength : int,
        maximum_ligation_window_length : int,
        time_unit : kdi.Unit,
    ):
    alphabet = ['A','T']
    motif_production_array = np.zeros((3,2,2,3)*2)
    complements = [1,0]
    strand_trajectory_id = '2023_07_26__09_57_42'
    motif_production_rate_constants = kdi.get.motif_production_rate_constants_from_strand_reactor_parameters(
        strand_trajectory_id,
        motiflength,
        complements,
        param_file_no = param_file_no,
        maximum_ligation_window_length = maximum_ligation_window_length
    )

    breakage_rate_constants = kdi.MotifBreakageVector(motiflength,alphabet,1./time_unit)
    breakage_vector_dct = kdi._create_empty_motif_breakage_dct(motiflength, alphabet)
    breakage_vector_array = np.zeros((3,2,2,3))
    breakage_rates = kdi.get.motif_breakage_rate_constants_from_strand_reactor_parameters(
        strand_trajectory_id,
        motiflength,
        complements,
        param_file_no = param_file_no
    )

    motif_vector_array = np.zeros((3,2,3,3))
    archive_path = kdi.utils.create_trajectory_ensemble_path(
        strand_trajectory_id=strand_trajectory_id,
        param_file_no=param_file_no,
        motiflength=motiflength,
    )
    makedirs(archive_path, exist_ok = True)
    strand_motif_trajectories = kdi.get.strand_motifs_trajectory_ensemble(
            motiflength,
            strand_trajectory_id,
            param_file_no,
            execution_time_path = archive_path
            )
    kdi.save_motif_trajectory_ensemble(archive_path+'sd/', strand_motif_trajectories)
    print(f"Archived motif trajectory in {archive_path+'sd/'}.")

    c_ref = kdi.get.strand_reactor_parameters(strand_trajectory_id)['c_ref']
    initial_motif_concentrations_vector = kdi.extract_initial_motif_vector_from_motif_trajectory(strand_motif_trajectories.trajectories[0], c_ref=c_ref)
    '''
    motif_vector_array[0,0,1,0] = 10./2459.*c_ref
    motif_vector_array[0,1,2,0] = 10./2459.*c_ref
    motif_vector_array[0,0,2,0] = 10./2459.*c_ref
    motif_vector_array[0,1,1,0] = 10./2459.*c_ref
    '''
    return alphabet, motif_production_rate_constants, breakage_rates, initial_motif_concentrations_vector, strand_motif_trajectories

def setup_zebra_0_scenario(
        motiflength : int,
        maximum_ligation_window_length : int,
        time_unit : kdi.Unit,
    ):
    return setup_zebra_i_scenario(0, motiflength, maximum_ligation_window_length, time_unit)

def setup_zebra_1_scenario(
        motiflength : int,
        maximum_ligation_window_length : int,
        time_unit : kdi.Unit,
    ):
    return setup_zebra_i_scenario(1, motiflength, maximum_ligation_window_length, time_unit)

def setup_zebra_2_scenario(
        motiflength : int,
        maximum_ligation_window_length : int,
        time_unit : kdi.Unit,
        zebra_fluctuation : float = 0.,
        braze_fluctuation : float = 0.,
        aa_fluctuation : float = 0.,
        bb_fluctuation : float = 0.,
        fourmer_fluctuation : float = 0.
    ):
    if (zebra_fluctuation+braze_fluctuation+aa_fluctuation+bb_fluctuation+fourmer_fluctuation) == 0.:
        return setup_zebra_i_scenario(2, motiflength, maximum_ligation_window_length, time_unit)
    alphabet, motif_production_rate_constants, breakage_rates, initial_motif_concentrations_vector, strand_motif_trajectories = setup_zebra_i_scenario(2, motiflength, maximum_ligation_window_length, time_unit)
    initial_motif_concentrations_vector = kdi.add_zebra_fluctuation_to_motif_vector(
        initial_motif_concentrations_vector,
        zebra_fluctuation=zebra_fluctuation,
        braze_fluctuation=braze_fluctuation,
        aa_fluctuation=aa_fluctuation,
        bb_fluctuation=bb_fluctuation,
        fourmer_fluctuation=fourmer_fluctuation
    )
    return alphabet, motif_production_rate_constants, breakage_rates, initial_motif_concentrations_vector, strand_motif_trajectories

def setup_zebra_2_fl(
        motiflength : int,
        maximum_ligation_window_length : int,
        time_unit : kdi.Unit,
        zebra_dimer_concentration : float = 0.,
        zebra_tetramer_concentration : float = 0.
    ):
    alphabet, motif_production_rate_constants, breakage_rates, initial_motif_concentrations_vector, strand_motif_trajectories = setup_zebra_i_scenario(2, motiflength, maximum_ligation_window_length, time_unit)
    initial_motif_concentrations_vector = kdi.convert_homogeneous_dimers_to_zebra_dimers(
        initial_motif_concentrations_vector,
        zebra_dimer_concentration = zebra_dimer_concentration
    )
    initial_motif_concentrations_vector = kdi.convert_dimers_to_tetramers(
        initial_motif_concentrations_vector,
        zebra_tetramer_concentration = zebra_tetramer_concentration
    )
    return alphabet, motif_production_rate_constants, breakage_rates, initial_motif_concentrations_vector, strand_motif_trajectories

def setup_zebra_3_scenario(
        motiflength : int,
        maximum_ligation_window_length : int,
        time_unit : kdi.Unit,
    ):
    return setup_zebra_i_scenario(3, motiflength, maximum_ligation_window_length, time_unit)

def setup_zebra_4_scenario(
        motiflength : int,
        maximum_ligation_window_length : int,
        time_unit : kdi.Unit,
    ):
    return setup_zebra_i_scenario(4, motiflength, maximum_ligation_window_length, time_unit)

if __name__=='__main__':
    jax.config.update('jax_enable_x64',True)
    # scenario = str(argv[-3]) # 'zebra_2' # 'zebra_1' # 'zebra_0' # 'zebra_2_2fl' # 'zebra_2_4fl' # 'LotkaVolterra' # 'Tkachenko' # 'monomer-dimer-system' # 'spontaneous_symmetrie_break' # 'Tkachenko' # '9999_99_99__99_99_99' #
    print(f'argv: {argv}')
    if len(argv)>3:
        scenario = str(argv[1])
    else:
        scenario = str(argv[-1])
    if scenario == '9999':
        scenario = '9999_99_99__99_99_99'
    print(f'Scenario: {scenario}')
    plotformats = ['.pdf']
    numbers_of_dimers = np.arange(-10.,10.+1.)
    numbers_of_tetramers = np.arange(-5., 5.+1.)
    annotation_style = {'xy':(0,1),'xycoords':'axes fraction', 'xytext':(+0.5,-0.5), 'textcoords':'offset fontsize', 'verticalalignment':'top'}

    if scenario[:5] == 'zebra':
        plotting_alphabet = ['X','Y']
    else:
        None


    if True:
        resolution_factor = None # 1e-2 #16*16 #1e-9
        t_span = (0,2.5e12)#(0,1.e12)
        complements = [1,0]
        motiflength = 4
        maximum_ligation_window_length = 4
        pseudo_count_concentration = 1.e-12
        pseudo_count_exp = None#-6#'-inf'#
        pseudo_zero_exp = -6
        if pseudo_count_exp=='-inf':
            pseudo_count = 0. 
        elif pseudo_count_exp is None:
            pseudo_count=None
        else:
            pseudo_count = 10.**pseudo_count_exp
        if pseudo_zero_exp=='-inf':
            pseudo_zero = 0.
        elif pseudo_zero_exp is None:
            pseudo_zero=None
        else:
            pseudo_zero = 10.**pseudo_zero_exp
            if pseudo_count is None:
                pseudo_count = 2*pseudo_zero
        soft_reactant_threshold = pseudo_count
        hard_reactant_threshold = pseudo_zero
        mass_correction_rate_constant_exp = -3#'-inf'
        mass_correction_rate_constant = 0. if mass_correction_rate_constant_exp == '-inf' else 10.**mass_correction_rate_constant_exp
        time_unit = kdi.read.symbol_config('time', unitformat=True)
        ode_integration_method = 'BDF'#'Dopri5'#'RK45'#
        print(f'ODE Integration Method: {ode_integration_method}')
        ivp_atol_exp = -10
        ivp_rtol_exp = -6
        ivp_atol = 10.**ivp_atol_exp
        ivp_rtol = 10.**ivp_rtol_exp
        parameters_string = f'md_{ode_integration_method}_{ivp_atol_exp}_{ivp_rtol_exp}_{mass_correction_rate_constant_exp}_{pseudo_count_exp}_{pseudo_zero_exp}'
        print(f"ivp_atol = 1e{ivp_atol_exp}, ivp_rtol = 1e{ivp_rtol_exp}, mass_correction_rate_constant = 1e{mass_correction_rate_constant_exp}")

        td_plot_parameters = {
                'linestyle' : '-.',
                'color' : kdi.plot.standard_colorbar()(1.),
                'alpha' : 1.,
                'label' : 'Theory',
                }
        sd_plot_parameters = {
                'linestyle' : '-.',
                'color' : kdi.plot.standard_colorbar()(0.5),
                'alpha' : 0.3,
                'label' : 'Strand',
                }
        md_plot_parameters = {
                'linestyle' : '--',
                'color' : kdi.plot.standard_colorbar()(0.),
                'alpha' : .3 if scenario in ['zebra_2_2fl','zebra_2_4fl'] else 1.,
                'label' : 'Motif',
                }


        strand_motif_trajectories = None
        if scenario == 'Tkachenko':
            alphabet, motif_production_rate_constants, breakage_rates, initial_motif_concentrations_vectors = setup_Tkachenko_scenario(
                motiflength,
                maximum_ligation_window_length,
                time_unit
            )
            initial_motif_concentrations_vectors = [initial_motif_concentrations_vectors,]
        elif scenario == 'monomer-dimer-system':
            alphabet, motif_production_rate_constants, breakage_rates, initial_motif_concentrations_vectors = setup_monomer_dimer_system(
                motiflength,
                maximum_ligation_window_length,
                time_unit
            )
            initial_motif_concentrations_vectors = [initial_motif_concentrations_vectors,]
        elif scenario == 'LotkaVolterra':
            alphabet, motif_production_rate_constants, breakage_rates, initial_motif_concentrations_vectors = setup_LotkaVolterra_system(
                motiflength,
                maximum_ligation_window_length,
                time_unit
            )
            initial_motif_concentrations_vectors = [initial_motif_concentrations_vectors,]
        elif scenario == 'spontaneous_symmetrie_break':
            alphabet, motif_production_rate_constants, breakage_rates, initial_motif_concentrations_vectors = setup_spontaneous_symmetry_break_scenario(
                motiflength,
                maximum_ligation_window_length,
                time_unit
            )
            initial_motif_concentrations_vectors = [initial_motif_concentrations_vectors,]
        elif scenario == '9999_99_99__99_99_99':
            alphabet, motif_production_rate_constants, breakage_rates, initial_motif_concentrations_vectors = setup_9999_scenario(
                motiflength,
                maximum_ligation_window_length,
                time_unit
            )
            initial_motif_concentrations_vectors = [initial_motif_concentrations_vectors,]
        elif scenario == 'zebra_0':
            alphabet, motif_production_rate_constants, breakage_rates, initial_motif_concentrations_vectors, strand_motif_trajectories = setup_zebra_0_scenario(
                motiflength,
                maximum_ligation_window_length,
                time_unit
            )
            initial_motif_concentrations_vectors = [initial_motif_concentrations_vectors,]

            strand_trajectory_id = '2023_07_26__09_57_42'
            cleavage_rate = kdi.get.strand_reactor_parameters(strand_trajectory_id)['r_delig']
            c_ref = kdi.get.strand_reactor_parameters(strand_trajectory_id)['c_ref']
            initial_motif_numbers_vector =  kdi.extract_initial_motif_vector_from_motif_trajectory(strand_motif_trajectories.trajectories[0])
            concentration_of_single_particle = c_ref/initial_motif_numbers_vector.motifs.val['length1strand'][0]
            onset_of_growth = kdi.infer.onset_of_growth(
                    np.sum(initial_motif_concentrations_vectors[0].motifs.val['length1strand']),
                    np.sum(initial_motif_concentrations_vectors[0].motifs.val['length2strand']),
                    np.max(kdi._motif_production_vector_as_array(motif_production_rate_constants)[0,:,:,0,0,:,:,0]),
                    np.max(kdi._motif_production_vector_as_array(motif_production_rate_constants)[0,:,:,1:,0,:,:,0]),
                    cleavage_rate
                    )
            discretized_onset_of_growth = kdi.infer.discretized_onset_of_growth(
                    np.sum(initial_motif_concentrations_vectors[0].motifs.val['length1strand']),
                    np.sum(initial_motif_concentrations_vectors[0].motifs.val['length2strand']),
                    np.max(kdi._motif_production_vector_as_array(motif_production_rate_constants)[0,:,:,0,0,:,:,0]),
                    cleavage_rate,
                    concentration_of_single_particle
                    )
            print('onset_of_growth')
            print(onset_of_growth)
            print('discretized_onset_of_growth')
            print(discretized_onset_of_growth)
        elif scenario == 'zebra_1':
            alphabet, motif_production_rate_constants, breakage_rates, initial_motif_concentrations_vectors, strand_motif_trajectories = setup_zebra_1_scenario(
                motiflength,
                maximum_ligation_window_length,
                time_unit
            )
            initial_motif_concentrations_vectors = [initial_motif_concentrations_vectors,]
        elif scenario == 'zebra_2':
            alphabet, motif_production_rate_constants, breakage_rates, initial_motif_concentrations_vectors, strand_motif_trajectories = setup_zebra_2_scenario(
                motiflength,
                maximum_ligation_window_length,
                time_unit
            )
            initial_motif_concentrations_vectors = [initial_motif_concentrations_vectors,]
        elif scenario == 'zebra_2_2fl':
            strand_trajectory_id = '2023_07_26__09_57_42'
            alphabet, motif_production_rate_constants, breakage_rates, _, strand_motif_trajectories = setup_zebra_2_fl(
                motiflength,
                maximum_ligation_window_length,
                time_unit,
                zebra_dimer_concentration = 0.,
                zebra_tetramer_concentration = 0.
            )
            c_ref = kdi.get.strand_reactor_parameters(strand_trajectory_id)['c_ref']
            initial_motif_numbers_vector =  kdi.extract_initial_motif_vector_from_motif_trajectory(strand_motif_trajectories.trajectories[0])
            concentration_of_single_particle = c_ref/initial_motif_numbers_vector.motifs.val['length1strand'][0]
            initial_motif_concentrations_vectors = [setup_zebra_2_fl(
                    motiflength,
                    maximum_ligation_window_length,
                    time_unit,
                    zebra_dimer_concentration = number_of_dimers*concentration_of_single_particle,
                    zebra_tetramer_concentration = 0.
                )[-2] for number_of_dimers in numbers_of_dimers]
            strand_motif_trajectories = None
        elif scenario == 'zebra_2_4fl':
            strand_trajectory_id = '2023_07_26__09_57_42'
            alphabet, motif_production_rate_constants, breakage_rates, _, strand_motif_trajectories = setup_zebra_2_fl(
                motiflength,
                maximum_ligation_window_length,
                time_unit,
                zebra_dimer_concentration = 0.,
                zebra_tetramer_concentration = 0.
            )
            c_ref = kdi.get.strand_reactor_parameters(strand_trajectory_id)['c_ref']
            initial_motif_numbers_vector =  kdi.extract_initial_motif_vector_from_motif_trajectory(strand_motif_trajectories.trajectories[0])
            concentration_of_single_particle = c_ref/initial_motif_numbers_vector.motifs.val['length1strand'][0]
            initial_motif_concentrations_vectors = [setup_zebra_2_fl(
                    motiflength,
                    maximum_ligation_window_length,
                    time_unit,
                    zebra_dimer_concentration = 0.,
                    zebra_tetramer_concentration = number_of_tetramers*concentration_of_single_particle
                )[-2] for number_of_tetramers in numbers_of_tetramers]
            strand_motif_trajectories = None
        elif scenario == 'zebra_3':
            alphabet, motif_production_rate_constants, breakage_rates, initial_motif_concentrations_vectors, strand_motif_trajectories = setup_zebra_3_scenario(
                motiflength,
                maximum_ligation_window_length,
                time_unit
            )
            initial_motif_concentrations_vectors = [initial_motif_concentrations_vectors,]
        elif scenario == 'zebra_4':
            alphabet, motif_production_rate_constants, breakage_rates, initial_motif_concentrations_vectors, strand_motif_trajectories = setup_zebra_4_scenario(
                motiflength,
                maximum_ligation_window_length,
                time_unit
            )
            initial_motif_concentrations_vectors = [initial_motif_concentrations_vectors,]
        else:
            raise NotImplementedError("Specified scenario '{}' does not exist.".format(scenario))

        times = None
        plotpath = './Plots/'+scenario+'/'+parameters_string+'/'
        makedirs(plotpath, exist_ok=True)

        if resolution_factor is not None:
            times = kdi.TimesVector(1./resolution_factor*np.arange(*(resolution_factor*t_span[0], resolution_factor*t_span[1]), dtype = np.float64), time_unit)
        else:
            times =  kdi.TimesVector(np.array([t_span[1]-t_span[0],], dtype = np.float64), time_unit)

        motif_production_log_rate_constants = kdi.MotifProductionVector(motiflength,alphabet,'1',maximum_ligation_window_length)(
                kdi._create_empty_motif_production_dict(
                    motiflength,
                    alphabet,
                    maximum_ligation_window_length
                    )
                )
        if scenario in ['zebra_0', 'zebra_1','zebra_2','zebra_2_2fl','zebra_2_4fl','zebra_3','zebra_4']:
            strand_trajectory_id = '2023_07_26__09_57_42'
            param_file_no= 2 if scenario in ['zebra_2_2fl','zebra_2_4fl'] else int(scenario[-1])
            archive_path = kdi.utils.create_trajectory_ensemble_path(
                strand_trajectory_id=strand_trajectory_id,
                param_file_no= 2 if scenario in ['zebra_2_2fl','zebra_2_4fl'] else int(scenario[-1]),
                motiflength=motiflength,
            )
            md_archive_path = archive_path + parameters_string
            md_archive_path += '_2fl'*(scenario=='zebra_2_2fl') + '_4fl'*(scenario=='zebra_2_4fl') + '/'
            print(f"{md_archive_path = }")

            if exists(md_archive_path):
                motif_trajectory_ensemble = kdi.load_motif_trajectory_ensemble(md_archive_path)
                motif_trajectory = motif_trajectory_ensemble.trajectories[0]
                times = motif_trajectory.times
                print(f"Loaded motif trajectory from {md_archive_path}.")
            else:
                makedirs(md_archive_path, exist_ok = True)
                motif_trajectories = [infer_fourmer_trajectory_from_rate_constants(
                    motif_production_rate_constants,
                    motif_production_log_rate_constants,
                    breakage_rates,
                    initial_motif_concentrations_vector,
                    times=times,
                    complements=complements,
                    mass_correction_rate_constant = mass_correction_rate_constant,
                    concentrations_are_logarithmized = False,
                    ode_integration_method = ode_integration_method,# 'BDF',#'Radau'
                    execution_time_path = md_archive_path + 'execution_time.txt',
                    pseudo_count_concentration = pseudo_count_concentration,
                    soft_reactant_threshold = soft_reactant_threshold,
                    hard_reactant_threshold = hard_reactant_threshold,
                    ivp_atol = ivp_atol,
                    ivp_rtol = ivp_rtol
                ) for initial_motif_concentrations_vector in initial_motif_concentrations_vectors]

                motif_trajectory_ensemble = kdi.MotifTrajectoryEnsemble(motif_trajectories)

                #archive trajectories
                kdi.save_motif_trajectory_ensemble(md_archive_path, motif_trajectory_ensemble)
                print(f"Archived motif trajectory in {md_archive_path}.")

            #plot trajectories
            if strand_motif_trajectories is not None:
                kdi.plot.motif_entropy(strand_motif_trajectories, **sd_plot_parameters)
        else:
            param_file_no = None
            motif_trajectories = [infer_fourmer_trajectory_from_rate_constants(
                motif_production_rate_constants,
                motif_production_log_rate_constants,
                breakage_rates,
                initial_motif_concentrations_vectors[0],
                times=times,
                complements=complements,
                mass_correction_rate_constant = mass_correction_rate_constant,
                concentrations_are_logarithmized = False,
                ode_integration_method = ode_integration_method,# 'BDF',#'Radau'
                execution_time_path = md_archive_path + 'execution_time.txt',
                pseudo_count_concentration = pseudo_count_concentration,
                soft_reactant_threshold = soft_reactant_threshold,
                hard_reactant_threshold = hard_reactant_threshold
            )]
            motif_trajectory = motif_trajectories[0]
            motif_trajectory_ensemble = kdi.MotifTrajectoryEnsemble(motif_trajectories)
        md_entropy_plot_params = md_plot_parameters.copy()
        md_entropy_plot_params['color'] = kdi.plot.greenish_colorbar()
        md_entropy_plot_params['alpha'] = 2./3.
        kdi.plot.motif_entropy(motif_trajectory_ensemble, **md_entropy_plot_params)
        plt.annotate('b',**annotation_style)
        plt.xscale('log')
        plt.yscale('log')
        if strand_motif_trajectories != None:
            xmax = np.max([tt for tt in [motif_trajectory_ensemble.times.val[-1], strand_motif_trajectories.times.val[-1]]])
        else:
            xmax = np.max([tt for tt in [motif_trajectory_ensemble.times.val[-1]]] + [t_span[-1]])
        plt.xlim(
                1e8,
                xmax
                )
        for plotformat in plotformats:
            plt.savefig(plotpath+f'entropy_{resolution_factor}_{ode_integration_method}'+plotformat)
            print(f"Saved {plotpath}entropy_{resolution_factor}_{ode_integration_method}{plotformat}")
        plt.close()

        # plot concentrations
        if strand_motif_trajectories is not None:
            for strand_motif_trajectory in strand_motif_trajectories.trajectories:
                plt.plot(strand_motif_trajectory.times.val, strand_motif_trajectory.motifs['length1strand'].val[:,0], **sd_plot_parameters)
        if scenario != 'monomer-dimer-system':
            for motif_trajectory in motif_trajectory_ensemble.trajectories:
                plt.plot(motif_trajectory.times.val, motif_trajectory.motifs['length1strand'].val[:,1], **md_plot_parameters)
                plt.plot(motif_trajectory.times.val, motif_trajectory.motifs['length2strand'].val[:,0,0], 'b-')
        if scenario != 'monomer-dimer-system':
            for motif_trajectory in motif_trajectory_ensemble.trajectories:
                plt.plot(motif_trajectory.times.val, motif_trajectory.motifs['length2strand'].val[:,0,1], 'g-')
                plt.plot(motif_trajectory.times.val, motif_trajectory.motifs['length2strand'].val[:,1,0], 'c-')
                plt.plot(motif_trajectory.times.val, motif_trajectory.motifs['length2strand'].val[:,1,1], 'r-')
        #plt.xlim(t_span[0],t_span[1])
        plt.xscale('log')
        plt.savefig(plotpath+f'concentrations_{resolution_factor}_{ode_integration_method}.pdf')
        plt.close()
        print("Saved {}".format(plotpath+f'concentrations_{resolution_factor}_{ode_integration_method}.pdf'))

        #plot mean length
        plt.close('all')
        mean_motif_strand_lengths = [(motif_trj.times.val, kdi.infer.mean_length(motif_trj)) for motif_trj in motif_trajectory_ensemble.trajectories]
        for mean_motif_strand_length in mean_motif_strand_lengths:
            plt.plot(*mean_motif_strand_length, **md_plot_parameters)
        if param_file_no is not None and scenario not in ['zebra_2_2fl','zebra_2_4fl']:
                try:
                    mean_strand_lengths = []
                    mean_length_archive_path = kdi.utils.create_strand_trajectory_ensemble_path(
                        strand_trajectory_id=strand_trajectory_id,
                        param_file_no= int(scenario[-1]),
                    )
                    mean_length_times_archive_path = mean_length_archive_path+'strand_mean_length_times_{}.npy'
                    mean_length_archive_path = mean_length_archive_path+'strand_mean_length_{}.npy'
                    for strand_length_trj_no in range(len(strand_motif_trajectories.trajectories)):
                        '''
                        with open(archive_path+'strand_mean_length.yaml', 'r') as yaml_file:
                            mean_length_properties = yaml.safe_load(yaml_file)
                        '''
                        mean_length_times = jnp.load(
                            mean_length_times_archive_path.format(strand_length_trj_no)
                        )
                        # mean_length_times = TimesVector(mean_length_times,mean_length_properties['times_unit'])
                        mean_lengths = jnp.load(
                            mean_length_archive_path.format(strand_length_trj_no)
                        )
                        mean_strand_lengths = mean_strand_lengths + [(mean_length_times, mean_lengths)]
                except FileNotFoundError:
                    mean_strand_lengths = [(strand_length_distribution['times'], strand_length_distribution['mean_length']) for strand_length_distribution in kdi.get.strand_length_distribution(strand_trajectory_id,param_file_no=param_file_no)]
                    for strand_length_trj_no in range(len(mean_strand_lengths)):
                        jnp.save(
                            mean_length_times_archive_path.format(strand_length_trj_no),
                            np.asarray(mean_strand_lengths[strand_length_trj_no][0])
                        )
                        jnp.save(
                            mean_length_archive_path.format(strand_length_trj_no),
                            np.asarray(mean_strand_lengths[strand_length_trj_no][1]),
                        )
                        '''
                        with open(archive_path+'strand_mean_length.yaml','w') as yaml_file:
                            yaml.dump({
                                'times_unit' : kdi.transform_unit_to_dict(strand_length_distribution['times'].domain[0].units)
                                },
                                yaml_file,
                                indent=4
                                )
                        '''
                for mean_strand_length in mean_strand_lengths:
                    plt.plot(*mean_strand_length, **sd_plot_parameters)
        if scenario in ['zebra_1','zebra_3']:
            plt.annotate('a',**annotation_style)
        elif scenario in ['zebra_2','zebra_4']:
            plt.annotate('b',**annotation_style)
        plt.xlim((10**9,None))
        plt.xscale('log')
        plt.xlabel(f"Time {kdi.transform_unit_to_str(time_unit)}")
        plt.ylabel('Mean Length [nucleotides]')
        for plotformat in plotformats:
            plt.savefig(plotpath+'mean_length'+plotformat)
            print(f'Saved {plotpath}mean_length{plotformat}.')

        plt.close('all')

        if strand_motif_trajectories is None:
            plotting_motif_trajectories = [motif_trajectory_ensemble,]
            plot_parameters = [md_plot_parameters]
            plot_zebraness_params = [md_entropy_plot_params]
        else:
            plotting_motif_trajectories = [strand_motif_trajectories, motif_trajectory_ensemble,]
            plot_parameters = [sd_plot_parameters, md_plot_parameters]
            plot_zebraness_params = [sd_plot_parameters, md_entropy_plot_params]

        kdi.plot.system_level_motif_zebraness(
            [motif_trajectory_ensemble,],
            plotting_time_windows=[[1e6,None]],
            plot_mean = [False,False],
            plotpath = plotpath+f'{resolution_factor}-md-',
            plot_parameters = plot_zebraness_params,
            plotformats = plotformats,
            annotation = {'text':'a'} | annotation_style,
        )
        kdi.plot.system_level_motif_zebraness(
            plotting_motif_trajectories,
            plotting_time_windows=[[1e6,None]],
            plot_mean = [False,]*len(plotting_motif_trajectories),
            plotpath = plotpath+f'{resolution_factor}-',
            plot_parameters = plot_zebraness_params,
            plotformats = plotformats,
            annotation = {'text':'a'} | annotation_style,
        )

        # calculate next Delta c
        """
        last_cv = motif_trajectory_ensemble.trajectories[0]...
        fourmer_production_rate_constants = 
        brc = 
        kdi.infer._integrate_motif_rate_equations(
                last_cv,
                number_of_letters=len(alphabet),
                motiflength=motiflength,
                complements=complements,
                concentrations_are_logarithmized=False,
                fourmer_production_rate_constants=fourmer_production_rate_constants,
                breakage_rate_constants = brc,
                )
        """

        c_ref = kdi.get.strand_reactor_parameters(strand_trajectory_id)['c_ref']
        if strand_motif_trajectories is not None:
            initial_motif_numbers_vector =  kdi.extract_initial_motif_vector_from_motif_trajectory(strand_motif_trajectories.trajectories[0])
            concentration_of_single_particle = c_ref/initial_motif_numbers_vector.motifs.val['length1strand'][0]
        else:
            concentration_of_single_particle = None
        for motifs_key in kdi.domains._return_motif_categories(motiflength):
            kdi.plot.motif_trajectories(
                plotting_motif_trajectories,
                alphabet = plotting_alphabet,
                plotting_time_windows = [[1e8,None]],
                plotpath = plotpath+f'{resolution_factor}-',
                c_ref = c_ref,
                ylim = [None if concentration_of_single_particle is None else concentration_of_single_particle/2,1.e-2],
                plot_parameters=plot_parameters,
                plotformats = plotformats,
                motifs_key = motifs_key,
                concentration_of_a_single_particle = concentration_of_single_particle,
            )

        for motif_trajectory in motif_trajectory_ensemble.trajectories:
            plt.plot(motif_trajectory.times.val, kdi.infer.total_mass_trajectory(motif_trajectory), **md_plot_parameters)
        plt.xlim((10**9,None))
        plt.xlabel(f"Time {kdi.transform_unit_to_str(time_unit)}")
        plt.ylabel('Total Mass [mol/L]')
        for plotformat in plotformats:
            plt.savefig(plotpath+f'total_mass_{resolution_factor}'+plotformat)
            print(f'Saved {plotpath}total_mass_{resolution_factor}{plotformat}')
        plt.close('all')

        if scenario != 'monomer-dimer-system':
            for motif_trajectory in motif_trajectory_ensemble.trajectories:
                plt.plot(motif_trajectory.times.val,motif_trajectory.motifs['length2strand'].val[:,0,0]-motif_trajectory.motifs['length2strand'].val[:,1,1], 'b-')
            plt.xscale('log')
            plt.savefig(plotpath+f'concentrations_dif_{resolution_factor}_r_linear-log.pdf')
            plt.xlim(
                t_span[0] if t_span[0]!=0. else 10**int(np.log10(motif_trajectory_ensemble.times.val[1])),
                t_span[1]
            )
            plt.ylim(-1.e-3,+1.e-3)
            plt.xscale('log')
            plt.close()
            print("Saved {}".format(plotpath+f'concentrations_dif_{resolution_factor}_r_linear-log.pdf'))
        gc.collect()
