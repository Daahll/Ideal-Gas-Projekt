#========================================================================================#
"""
	IdealGas
änderung
Extending SimpleParticles to conserve kinetic energy and momentum.

Author: Francisco Hella, Felix Rollbühler, Melanie *, Jan Wiechmann, 22/06/23
"""
module IdealGas

include("AgentTools.jl")
include("TD_Physics.jl")
using Agents, LinearAlgebra, GLMakie, InteractiveDynamics, .AgentTools

#-----------------------------------------------------------------------------------------
# Module types:
#-----------------------------------------------------------------------------------------
"""
	Particle

The populating agents in the IdealGas model.
"""
@agent Particle ContinuousAgent{2} begin
	mass::Float64					# Particle's mass
	speed::Float64					# Particle's speed
	radius::Float64					# Particle's radius
	prev_partner::Int				# Previous collision partner id
	last_bounce::Float64			
end

"Standard value that is definitely NOT a valid agent ID"
const non_id = -1

#-----------------------------------------------------------------------------------------
# Module methods:
#-----------------------------------------------------------------------------------------
"""
	idealgas( kwargs)

Create and initialise the IdealGas model.
"""
function idealgas(;
	width = 500,
	gases = Dict("Helium" => 4.0, "Hydrogen" => 1.0, "Oxygen" => 32.0),					# Gas types
	total_volume = 5.0,																	# Initial volume of the container
	volume = calc_total_vol_dimension(total_volume), 									# Dimensions of the container
	topBorder = total_volume/5.0,
	temp = 293.15,																		# Initial temperature of the gas in Kelvin
	temp_old = 293.15,
	pressure_bar = 1.0,																	# Initial pressure of the gas in bar
	pressure_pa =  pressure_bar*1e5,													# Initial pressure of the gas in Pascal
	n_mol = pressure_pa * volume[1] * volume[2] * volume[3] / (8.314*temp),				# Number of mol
	init_n_mol = copy(n_mol), 															# Initial number of mol
	real_n_particles = n_mol * 6.022e23/4,												# Real number of Particles in box: Reduction for simplicity
	real_n_particles = n_mol * 6.022e23/4,												# Real number of Particles in box: Reduction for simplicity
    n_particles = real_n_particles/1e23,												# Number of Particles in simulation box
	molare_masse = 4.0,																	# Helium Gas mass in atomic mass units
	mass_u = 4.0,
	mass_kg = molare_masse * 1.66053906660e-27,											# Convert atomic/molecular mass to kg
	mass_gas = round(n_mol * molare_masse, digits=3),									# Mass of gas
	radius = 4.0,																		# Radius of Particles in the box
	# U = 3/2 * N(Anzahl Part) * k(Boltzmann) * T = 3/2 * n(mol) * R * T
	e_internal = 3/2 * n_mol * 8.314 * temp,											# Inner energy of the gas
	entropy_change = 0.0,																# Change in entropy of the gas
	extent = (width,width),																	# Extent of Particles space
)
    space = ContinuousSpace(extent; spacing = radius/2.0)

	properties = Dict(
		:n_particles		=> n_particles,
		:temp				=> temp,
		:temp_old			=> temp_old,
		:total_volume		=> total_volume,
		:e_internal			=> e_internal,
		:entropy_change 	=> entropy_change,
		:pressure_pa		=> pressure_pa,
		:pressure_bar		=> pressure_bar,
		:real_n_particles	=> real_n_particles,
		:n_mol		=> n_mol,
		:volume	=> volume,
		:temp_old	=> temp_old,
		:pressure_bar_old	=> pressure_bar_old,
		:init_n_mol	=> init_n_mol,
		:gases		=> gases,
		:molare_masse		=> molare_masse,
		:mass_kg		=> mass_kg,
		:mass_gas	=> mass_gas,
		:step => 0,
		:cylinder_command => 0, 
		:cylinder_pos => 500,
		:reduce_volume_merker => 500,
		:modes				=> modes,
		:mode				=> mode,
		:objective 			=> create_heatmap(width),
	)


    box = ABM( Particle, space; properties, scheduler = Schedulers.Randomly())

	molare_masse_kg = box.molare_masse / 1000	# Convert g/mol to kg/mol
	max_speed = 4400.0  # Maximum speed in m/s
	molare_masse_kg = box.molare_masse / 1000	# Convert g/mol to kg/mol
	max_speed = 4400.0  # Maximum speed in m/s
	for _ in 1:n_particles
		vel = Tuple( 2rand(2).-1)
		vel = vel ./ norm(vel)  # ALWAYS maintain normalised state of vel!
		speed = sqrt((3 * k * box.temp) / mass_kg)  # Initial speed based on temperature
		speed = TD_Physics.scale_speed(speed, max_speed)  		# Scale speed to avoid excessive velocities
		#speed = scale_speed(speed, max_speed)  		# Scale speed to avoid excessive velocities
        add_agent!( box, vel, mass_kg, speed, radius, non_id, -Inf)
	end
    return box
end
#-----------------------------------------------------------------------------------------
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
	create_heatmap(width)

		Creates the underlying heatmap as a representation of the gas-tank.
"""
	function create_heatmap(width::Int)
		outer_rim = 1
		heatarray = zeros(width, width)  # Initialize the heatarray with zeros
	
		map((i) -> begin
			x = i[1]
			y = i[2]
			if (x >= 1 && x <= outer_rim) || (y >= 1 && y <= outer_rim) || (x <= width && x >= width - outer_rim) || (y <= width && y >= width - outer_rim)
				heatarray[x, y] = 1.0
			end
		end, CartesianIndices(heatarray))
	
		return heatarray
	end
#-----------------------------------------------------------------------------------------
"""
	agent_step!( me, box)

This is the heart of the IdealGas model: It calculates how Particles collide with each other,
while conserving momentum and kinetic energy.
"""
function agent_step!(me::Particle, box::ABM)
	her = random_nearby_agent( me, box, 2*me.radius)	# Grab nearby particle
	if her === nothing
		# No new partners - forget previous collision partner:
		me.prev_partner = non_id
	elseif her.id < me.id && her.id != me.prev_partner
		# New collision partner has not already been handled and is not my previous partner:
		me.prev_partner = her.id							# Update previous partners to avoid
		her.prev_partner = me.id							# repetitive juddering collisions.
		cntct = (x->[cos(x),sin(x)])(2rand()pi)				# Unit vector to contact point with partner
		Rctct = [cntct[1] cntct[2]; -cntct[2] cntct[1]]		# Rotation into contact directn coords
		Rback = [cntct[1] -cntct[2]; cntct[2] cntct[1]]		# Inverse rotation back to world coords

		# Rotate velocities into coordinates directed ALONG and PERPendicular to contact direction:
		myAlongVel, myPerpVel = me.speed * Rctct * collect(me.vel)					# My velocity
		herAlongVel, herPerpVel = her.speed * Rctct * collect(her.vel)				# Her velocity
		cmAlongVel = (me.mass*myAlongVel + her.mass*herAlongVel)/(me.mass+her.mass)	# C of M velocity

		# Calculate collision effects along contact direction (perp direction is unaffected):
		myAlongVel = 2cmAlongVel - myAlongVel
		herAlongVel = 2cmAlongVel - herAlongVel

		# Rotate collision effects on both me and her back into world coordinates:
		me.speed = hypot(myAlongVel,myPerpVel)
		if me.speed != 0.0
			me.vel = Tuple(Rback*[myAlongVel,myPerpVel])
			me.vel = me.vel ./ norm(me.vel)
		end
		her.speed = hypot(herAlongVel,herPerpVel)
		if her.speed != 0.0
			her.vel = Tuple(Rback*[herAlongVel,herPerpVel])
			her.vel = her.vel ./ norm(her.vel)
		end
	end

	check_particle_near_border!(me, box)  # Aufruf der neuen Funktion

	# Zylinder Steuerug 

	if box.cylinder_command == 1
		button_reduce_volume!(me, box)
		#println("zylinder fährt ein")
	elseif box.cylinder_command == 0
		box.reduce_volume_merker = box.cylinder_pos
	elseif box.cylinder_command == 2
		button_increase_volume!(me,box)
		box.reduce_volume_merker = box.space.extent[1] # solange Funtkion aktiv wird die Grenze bei check_particle_near_border aufgehoben 
	end 
	
	#println("vor move agent: ")
	#println(me.speed)
	move_agent!(me, box, me.speed)
	#println("nach move agent: ")
	#println(me.speed)
end
#----------------------------------------------------------------------------------------

function check_particle_near_border!(me, box)
    x, y = me.pos

    if x < 1.8 + me.radius/2 && box.step - me.last_bounce > 3
        me.vel = (-me.vel[1], me.vel[2])
        me.last_bounce = box.step
    elseif x > box.reduce_volume_merker - 1.8 && box.step - me.last_bounce > 3
        me.vel = (-me.vel[1], me.vel[2])
        me.last_bounce = box.step
    end
    if y < 1.8 + me.radius/2 && box.step - me.last_bounce > 3
        me.vel = (me.vel[1], -me.vel[2])
        me.last_bounce = box.step			
    elseif y > box.space.extent[2] - 1.8 && box.step - me.last_bounce > 3 
        me.vel = (me.vel[1], -me.vel[2])
        me.last_bounce = box.properties[:step]
    end

end

#-----------------------------------------------------------------------------------------
function button_increase_volume!(me, box)

	x,y = me.pos

	println(box.cylinder_pos)

	if box.cylinder_pos > 499.5 
    	println("zylinder ist in Ursprunngsposition")
	
	elseif x > box.cylinder_pos 
			me.vel = (-me.vel[1], me.vel[2])	
			me.speed = me.speed/2 # Anahme: die Hälfte der Energie wird abgegeben 
	end 
	
end


#-----------------------------------------------------------------------------------------
"""
	model_step!( model)

	calculate the quantities, based on the chosen mode (Specifies which variables are constant)
	calculate the quantities, based on the chosen mode (Specifies which variables are constant)
"""
function model_step!(model::ABM)

	
	if model.mode == "temp-druck" || model.mode == "vol-druck"
		model.volume = calc_total_vol_dimension(model.total_volume)
		pressure_pa = model.n_mol * 8.314 * model.temp / (model.volume[1] * model.volume[2] * model.volume[3])
		model.pressure_pa = round(pressure_pa, digits=3)
		model.pressure_bar = round(model.pressure_pa / 1e5, digits=2)
	elseif model.mode == "temp-vol" || model.mode == "druck-vol"
		model.total_volume = model.n_mol * 8.314 * model.temp/ model.pressure_pa
		model.volume = calc_total_vol_dimension(model.total_volume)
		model.topBorder = model.total_volume/5.0
	elseif model.mode == "druck-temp" || model.mode == "vol-temp"
		model.volume = calc_total_vol_dimension(model.total_volume)
		temp = calc_temperature(model)
		model.temp = round(temp, digits=2)
	end

	model.entropy_change = calc_entropy_change(model)
	
	model.e_internal = calc_internal_energy(model)

	molare_masse_kg = model.molare_masse / 1000	# Convert g/mol to kg/mol
	max_speed = 4400.0  # Maximum speed in m/s
	u_rms = sqrt((3 * R * model.temp) / molare_masse_kg)  # Root mean squared speed based on temperature
	for particle in allagents(model)
		particle.speed = TD_Physics.scale_speed(u_rms, max_speed)  
	end

	model.step += 1.0


	if model.cylinder_command == 1 && model.cylinder_pos > 250 # Zylinder soll ausgefahren werden
		model.cylinder_pos = model.cylinder_pos - 0.3
		#println("volumen wird veringert")
	elseif model.cylinder_command == 2 && model.cylinder_pos < 500 # Zylinder soll zurück gefahren werden
		model.cylinder_pos = model.cylinder_pos + 0.3 
		#println("volume wird erhöht")
	end
end

#----------------------------------------------------------------------------------------

"""
	demo()

Run a simulation of the IdealGas model.
"""
	function demo()
		box = idealgas()

		
		plotkwargs = (;
    		ac = :skyblue3,
    		scatterkwargs = (strokewidth = 1.0,),
			as = 8.0,
			add_colorbar = false,
			colormap=:greys,
			colorrange=(0, 1),
			heatarray=:objective,
		)
	
		entropy(box) = box.entropy_change
		mdata = [entropy]
		mlabels = ["ΔS in [J/K] (Entropieänderung)"]
	
		playground,abmobs = abmplayground( box, idealgas;
			agent_step!,
			model_step!,
			mdata,
			mlabels,
			figure = (;
						xlabel = "width in cm", 
						ylabel ="height in cm", 
						resolution = (1300, 750)
					),
			plotkwargs...
		)

		# Figure Objekten neues Layout zuweisen durch feste Reihenfolge in figure.content[i]
		model_plot = playground.content[1]	# Box 	
		playground[0:2,0] = model_plot
		entropy_plot = playground.content[7]
		playground[0:1,2][1,0:1] = entropy_plot
		# Sliders
		playground[2,1] = playground.content[2]
		playground[2,2] = playground.content[7]
		# Buttons
		gl_buttons = playground[3,1] = GridLayout()
		gl_buttons[0,2] = playground.content[3]
		gl_buttons[0,3] = playground.content[4]
		gl_buttons[0,4] = playground.content[5]
		gl_buttons[0,5] = playground.content[6]

		gl_sliders = playground[4,:] = GridLayout()
		gl_dropdowns = playground[3,0] = GridLayout()
		gl_labels = playground[0,1] = GridLayout()

		gas_dropdown = Menu(gl_dropdowns[0,0], options = keys(box.gases), default = "Helium")
		mode_dropdown = Menu(gl_dropdowns[0,1], options = keys(box.modes), default = "Temperatur:Druck")

		pressure_label = Label(gl_labels[2,0], "Druck: " * string(round(box.pressure_bar, digits=2))* " Bar", fontsize=22)
		mass_label = Label(gl_labels[3,0], "Masse: " * string(box.mass_gas)* " g", fontsize=22)
		volume_label = Label(gl_labels[1,0], "Volumen: " * string(round(box.total_volume, digits=2))* " m³ ; " * string(round(box.total_volume * 1000, digits=2)) * " L", fontsize=22)
		e_internal_label = Label(gl_labels[4,0], "Eᵢ: " * string(round(box.e_internal, digits=2)) * " J", fontsize=22)

		#Custom Buttons
		increase_vol_btn = Button(vol_change_btns[0,1:2], label = "Increase\nVolumen")# = print("increase"))#increase_vol_const())
		pause_vol_btn = Button(vol_change_btns[0,3], label = "Pause")
		decrease_vol_btn = Button(vol_change_btns[0,4:5], label = "Decrease\nVolumen")# = print("decrease"))#decrease_vol_const())

	
		#TODO: Hier volumen change funktionen aufrufen
		on(increase_vol_btn.clicks) do _
			box.cylinder_command = 1
		end  

		on(pause_vol_btn.clicks) do _
			box.cylinder_command = 0
		end 

		on(decrease_vol_btn.clicks) do _
			box.cylinder_command = 2
		end


		# Custom Slider
		# Allows to set the value of the slider
		# Allows to prevent value change when the slider is moved
		temp_slider_label = Label(slider_space[0,0], "Temperatur: ", fontsize=16)
		temp_slider = Slider(slider_space[0,1], range = 0.0:0.01:1000.0, startvalue=293.15)
		temp_slider_value = Label(slider_space[0,2], string(temp_slider.value[]) * " K")


		pressure_slider_bar_label = Label(slider_space[1,0], "Druck: ", fontsize=16)
		pressure_slider_bar = Slider(slider_space[1,1], range = 0.1:0.1:10.0, startvalue=1.0)
		pressure_slider_bar_value = Label(slider_space[1,2], string(pressure_slider_bar.value[]) * " Bar")

		pressure_slider_pa_label = Label(slider_space[2,0], "Druck: ", fontsize=16)
		pressure_slider_pa = Slider(slider_space[2,1], range = 1.0:1.0:1000000.0, startvalue=100000.0)
		pressure_slider_pa_value = Label(slider_space[2,2], string(pressure_slider_pa.value[]) * " Pa")

		volume_slider_label = Label(slider_space[3,0], "Volumen: ", fontsize=16)
		volume_slider = Slider(slider_space[3,1], range = 0.1:0.1:30.0, startvalue=2.0)
		volume_slider_value = Label(slider_space[3,2], string(volume_slider.value[]) * " m³")


		on(abmobs.model) do _
			e_internal_label.text[] = string("Eᵢ: ", string(round(box.e_internal)), " J")
			pressure_label.text[] = string("Druck: ", string(round(box.pressure_bar, digits=2)), " Bar")
			if box.mass_gas > 999.9
				mass_label.text[] = string("Masse: ", string(round(box.mass_gas/1000, digits=3), " kg"))
			else
				mass_label.text[] = string("Masse: ", string(box.mass_gas), " g")
			end
			volume_label.text[] = string("Volumen: " * string(round(box.total_volume, digits=2))* " m³ ; " * string(round(box.total_volume * 1000, digits=2)) * " L")

		end

		on(gas_dropdown.selection) do selected_gas
			new_molare_masse = box.gases[selected_gas]
			box.molare_masse = new_molare_masse
			box.mass_kg = new_molare_masse * 1.66054e-27
			box.mass_gas = round(box.n_mol * box.molare_masse, digits=3)
			
		end

		on(mode_dropdown.selection) do selected_mode
			box.mode = box.modes[selected_mode]
		end

		on(temp_slider.value) do temp
			if box.mode == "temp-druck" || box.mode == "temp-vol"
				temp_slider_value.text[] = string(temp[]) * " K"
				box.temp = temp[]
			end
			
			if box.mode == "temp-druck"
				pressure = calc_pressure(box)
				pressure_slider_pa_value.text[] = string(round(pressure, digits=0)) * " Pa"
				box.pressure_pa = pressure
				set_close_to!(pressure_slider_pa, pressure)
				pressure_slider_bar_value.text[] = string(round(pressure / 1e5, digits=2)) * " Bar"
				box.pressure_bar = pressure / 1e5
				set_close_to!(pressure_slider_bar, pressure / 1e5)
			elseif box.mode == "temp-vol"
				volume = box.n_mol * 8.314 * box.temp/ box.pressure_pa
				volume_slider_value.text[] = string(round(volume, digits=2)) * " m³"
				set_close_to!(volume_slider, volume)
			end
		end

		on(pressure_slider_bar.value) do pressure
			if box.mode == "druck-vol" || box.mode == "druck-temp"
				pressure_slider_bar_value.text[] = string(round(pressure[], digits=2)) * " Bar"
				box.pressure_bar = pressure[]

				pressure_slider_pa_value.text[] = string(round(pressure[] * 1e5, digits=0)) * " Pa"
				box.pressure_pa = pressure[] * 1e5
				
				if box.mode == "druck-vol"
					volume = box.n_mol * 8.314 * box.temp/ box.pressure_pa
					volume_slider_value.text[] = string(round(volume, digits=2)) * " m³"
					set_close_to!(volume_slider, volume)
				elseif box.mode == "druck-temp"
					temp = calc_temperature(box)
					temp_slider_value.text[] = string(round(temp, digits=2)) * " K"
					set_close_to!(temp_slider, temp)
				end

			end
		end

		on(pressure_slider_pa.value) do pressure
			if box.mode == "druck-vol" || box.mode == "druck-temp"
				pressure_slider_pa_value.text[] = string(round(pressure[], digits=0)) * " Pa"
				box.pressure_pa = pressure[]

				pressure_slider_bar_value.text[] = string(round(pressure[] / 1e5, digits=2)) * " Bar"
				box.pressure_bar = pressure[] / 1e5

				if box.mode == "druck-vol"
					volume = box.n_mol * 8.314 * box.temp/ box.pressure_pa
					volume_slider_value.text[] = string(round(volume, digits=2)) * " m³"
					set_close_to!(volume_slider, volume)
				elseif box.mode == "druck-temp"
					temp = calc_temperature(box)
					temp_slider_value.text[] = string(round(temp, digits=2)) * " K"
					set_close_to!(temp_slider, temp)
				end

			end
		end

		on(volume_slider.value) do volume
			if box.mode == "vol-druck" || box.mode == "vol-temp"
				volume_slider_value.text[] = string(round(volume[], digits=2)) * " m³"
				box.total_volume = volume[]

				if box.mode == "vol-druck"
					pressure = calc_pressure(box)
					pressure_slider_pa_value.text[] = string(round(pressure, digits=0)) * " Pa"
					box.pressure_pa = pressure[]
					set_close_to!(pressure_slider_bar, pressure[] / 1e5)
					pressure_slider_bar_value.text[] = string(round(pressure / 1e5, digits=2)) * " Bar"
					box.pressure_bar = pressure / 1e5
					set_close_to!(pressure_slider_pa, pressure)
				elseif box.mode == "vol-temp"
					temp = calc_temperature(box)
					temp_slider_value.text[] = string(round(temp, digits=2)) * " K"
					set_close_to!(temp_slider, temp)
				end
			end
		end

		playground
	end
end	# of module IdealGas