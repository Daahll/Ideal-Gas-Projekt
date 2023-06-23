#========================================================================================#
"""
TD_Physics

Custom Library of ThermoDynamics for IdealGas-simulation.

Author: Francisco Hella, Felix Rollbühler, Melanie Heinrich, Jan Wichmann, 20/06/23
"""

module TD_Physics

import Agents

export calc_temperature, calc_pressure, calc_n_mol, calc_real_n_particles, momentum, kinetic_energy, scale_speed, calc_total_vol_dimension

#-----------------------------------------------------------------------------------------
"""
	momentum( particle)

Return the momentum of this particle.
"""
function momentum(particle)
	particle.mass * particle.speed * collect(particle.vel)
end
#-----------------------------------------------------------------------------------------
"""
	kinetic_energy( particle)

Return the kinetic energy of this particle.
"""
	function kinetic_energy(particle)
		particle.mass * particle.speed^2 / 2
	end

#-----------------------------------------------------------------------------------------
"""
	scale_speed(speed, max_speed)

Scales a speed value to the interval [0,1] based on the provided max_speed.
"""
	function scale_speed(speed, max_speed)
		if speed > max_speed
			speed = max_speed
		end
		return speed / max_speed
	end

#-----------------------------------------------------------------------------------------
"""
	calc_n_mol(model)

Return the number of molecules in the system.
"""
	function calc_n_mol(model)
		return model.pressure_bar * 1e5 * model.total_volume / (8.314*model.temp)
	end

#------------------------------------------------------------------------------------------
"""
	calc_real_n_particles(model)

Return the number of particles in the system.
"""
	function calc_real_n_particles(model)
		return model.n_mol * 6.022e23
	end

#------------------------------------------------------------------------------------------
"""
calc_total_vol_dimension( me, box)

Calculates volume/dimension of a 3D-Space with [x, y=5, z=1], based on a given value of total volume.
"""
function calc_total_vol_dimension(volume, x_axis_vol=5.0)
 	y_axis_vol = volume/x_axis_vol
 	return [y_axis_vol, x_axis_vol, 1.0] 
end
#-----------------------------------------------------------------------------------------
"""
	calc_pressure(model)

Return the pressure of the system.
"""
	function calc_pressure(model)
		R = 8.314 # Gaskonstante in J/(mol·K)
		n = model.n_mol # Anzahl der Moleküle (angenommen, jedes Partikel repräsentiert ein Molekül)
		V = model.total_volume # Volumen des Behälters
		T = model.temp # Durchschnittstemperatur der Moleküle
		P = n * R * T / V
		return P
	end

end #------------------------------------------------------------------------------------------
"""
    calc_volume(model)

Return the volume of the system.
"""
function calc_volume(model)
    R = 8.314 # Gaskonstante in J/(mol·K)
    n = model.n_mol # Anzahl der Moleküle (angenommen, jedes Partikel repräsentiert ein Molekül)
    T = model.temp # Durchschnittstemperatur der Moleküle
    P = model.pressure_pa
    V = n * R * T / P
    return V
end

#---------------------------------------------------------------------------------------------
"""
	calc_temperature

Return the temperature of the system.
"""
	function calc_temperature(model)   
		# T = Ekin / (k * 2/3 * N ); Boltzmann constant k = 1.38e-23
		P = model.pressure_pa
		R = 8.314 # Gaskonstante in J/(mol·K)
		n = model.n_mol # Anzahl der Moleküle
		V = model.total_volume # Volumen des Behälters
		T = (P*V)/(R*n)
		return T
	end

#------------------------------------------------------------------------------------------