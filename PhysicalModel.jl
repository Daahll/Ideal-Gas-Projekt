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

Calculate the specific heat capacity and return the change in entropy of the gas depening on the thermodynamic process.
"""
function calc_entropy_change(model) 
	mass_gas_kg = model.mass_gas / 1000											# Convert g to kg			
	molar_mass_kg = model.molar_mass / 1000										# Convert g/mol to kg/mol
	R_i = round((R / molar_mass_kg), digits=3)									# Individual gas constant

	# Isochoric process or Isochoric & isothermal process
	if model.mode == "druck-temp" || model.mode == "temp-druck"	|| model.mode == "mol-druck" 
		c_mp = (model.f + 2)* R/2													# Molar heat capacity at constant pressure in [J/molK]
		# Round every parameter to 3 digits
		c_p = round(c_mp, digits=3) * round(model.n_mol, digits=3) / mass_gas_kg 	# Specific heat capacity cₚ in [J/kgK]
		# Δs = cₚ · ln(T₂/T₁) + Rᵢ · ln(p₂/p₁) 			
		Δs = c_p * log(model.temp/model.temp_old) + R_i * log(model.pressure_pa/model.pressure_pa_old)

	elseif model.mode == "vol-temp"	|| model.mode == "vol-druck" 					# Isobaric process or isothermal process
		c_mv = model.f * R/2 														# Molar heat capacity at constant volume 
		c_v = round(c_mv, digits=3) * round(model.n_mol, digits=3) / mass_gas_kg	# Specific heat capacity cᵥ
		# Δs = cᵥ · ln(T₂/T₁) + Rᵢ · ln(V₂/V₁)			
		Δs = c_v * log(model.temp/model.temp_old) + R_i * log(model.total_volume_m3/model.total_volume_m3_old)
	
	elseif model.mode == "mol-temp"
		Δs = 0.0 # Entropy change calculation missing for mode:mol-temp
	end
	
	model.temp_old = model.temp							# Set the system variables used in the next step
	model.total_volume_m3_old = model.total_volume_m3

	return Δs
end

#------------------------------------------------------------------------------------------
"""
	calc_internal_energy

Return the internal energy of the system.
"""
function calc_internal_energy(model)  
	model.f * 1/2 * model.n_mol * R * model.temp 		# Eᵢ = f * 1/2 * n * R * T 	
end

#------------------------------------------------------------------------------------------
"""
	calc_and_scale_speed(model)

Return the scaled root mean squared speed of the particles based on temperature.
"""
function calc_and_scale_speed(model)  
	max_speed = 5000									# Maximum speed in m/s; Cap the visual speed at about T=3700 K, V=250 L, p=4 bar, n=4 mol
	molar_mass_kg = model.molar_mass / 1000				# Convert g/mol to kg/mol
	speed = sqrt((3 * R * model.temp) / molar_mass_kg)  # Root mean squared speed based on temperature uᵣₘₛ = sqrt(3*R*T / M)
	scaled_speed = PhysicalModel.scale_speed(speed, max_speed)  	# Scale speed to avoid excessive velocities
	return scaled_speed
end

#------------------------------------------------------------------------------------------

end # module