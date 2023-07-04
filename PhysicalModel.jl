<#========================================================================================#
"""
PhysicalModel

Model of all the physical calculations for the thermodynamic simulation.

Author: Francisco Hella, Felix Rollbühler, Melanie *, Jan Wiechmann, 20/06/23
"""

module PhysicalModel

include("AgentTools.jl")
using Agents, LinearAlgebra, GLMakie, InteractiveDynamics, GeometryBasics, Observables

export calc_temperature, calc_pressure, calc_n_mol, calc_real_n_particles, momentum, kinetic_energy, scale_speed, calc_and_scale_speed, calc_total_vol_dimension, calc_entropy_change, calc_internal_energy, calc_volume

const R = 8.314
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
	scale_speed(speed, max_speed)

Scales a speed value to the interval [0,1] based on the provided max_speed.
"""
function scale_speed(speed, max_speed)
	if speed > max_speed
		speed = max_speed
	end
	return 15 * speed / max_speed
end
#------------------------------------------------------------------------------------------
"""
	calc_n_mol(model)

Return the number of molecules in the system.
"""
function calc_n_mol(model)
	return model.pressure_bar * 1e5 * model.total_volume_m3 / (8.314*model.temp)
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
	n = model.n_mol # Anzahl der Moleküle (angenommen, jedes Partikel repräsentiert ein Molekül)
	V = model.total_volume_m3 # Volumen des Behälters
	T = model.temp # Durchschnittstemperatur der Moleküle
	P = n * R * T / V
	return P
end

#------------------------------------------------------------------------------------------
"""
    calc_volume(model)

Return the volume of the system.
"""
function calc_volume(model)
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
	P = model.pressure_pa
	n = model.n_mol # Anzahl der Moleküle
	V = model.total_volume_m3
	T = (P*V)/(R*n)
	return T
end

#------------------------------------------------------------------------------------------
"""
	calc_entropy_change

Return the change in entropy of the system.
"""
function calc_entropy_change(model)   
	# Specific heat capacity depending on the thermodynamic process 
	if model.mode == "druck-temp" || model.mode == "temp-druck"	# Isochor
		# Cᵥ = 3/2 * R (monoatomic)												# TODO:Freiheitsgrade miteinbeziehen?-> Cᵥ = 5/2 * R (diatomic) 
		C = 3/2 * R
	elseif model.mode == "vol-temp" || model.mode == "temp-vol"	# Isobar						
		# Cₚ = 5/2 * R (monoatomic)												# TODO: Cₚ = 7/2 * R (diatomic)	
		C = 5/2 * R
	else				# Isothermal process: No change in entropy
		return 0.0
	end
	Δtemp = model.temp - model.temp_old			# ΔT = T₂ - T₁
	ΔQ = model.n_mol * C * Δtemp				# ΔQ = n * C * ΔT (Heat Exchange)
	ΔS = ΔQ / model.temp						# ΔS = ΔQ / T	(Change in Entropy)
	model.temp_old = model.temp
	return ΔS
end

#------------------------------------------------------------------------------------------
"""
	calc_internal_energy

Return the internal energy of the system.
"""
function calc_internal_energy(model)  
	# Eᵢ = 3/2 * n * R * T (monoatomic) 
	3/2 * model.n_mol * R * model.temp 			#TODO: Eᵢ = 5/2 * n * R * T (diatomic)
end

#------------------------------------------------------------------------------------------
"""
	calc_and_scale_speed(model)

Return the scaled root mean squared speed of the particles based on temperature.
"""
function calc_and_scale_speed(model)  
	max_speed = 5000 										# Maximum speed in m/s; Cap the visual speed at about T=1450 K, V=250L, p=4 bar
	molare_masse_kg = model.molar_mass / 1000				# Convert g/mol to kg/mol
	speed = sqrt((3 * R * model.temp) / molare_masse_kg)  	# Root mean squared speed based on temperature uᵣₘₛ = sqrt(3*R*T / M)
	scaled_speed = PhysicalModel.scale_speed(speed, max_speed)  			# Scale speed to avoid excessive velocities
	return scaled_speed
end

#------------------------------------------------------------------------------------------

end # module
