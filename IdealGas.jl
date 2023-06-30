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
const R = 8.314 # Gaskonstante in J/(mol·K)

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
	modes = Dict("Temperatur:Druck" => "temp-druck",
				 "Temperatur:Volumen" => "temp-vol",
				 "Druck:Temperatur" => "druck-temp",
				 "Druck:Volumen" => "druck-vol",
				 "Volumen:Temperatur" => "vol-temp",
				 "Volumen:Druck" => "vol-druck",									
				 "Mol : Temperatur" => "mol-temp",
				 "Mol : Druck" => "mol-druck",
				),										# Modes of operation
	mode = "temp-druck",
	total_volume = 0.2,																	# Initial volume of the container
	volume = calc_total_vol_dimension(total_volume),
	temp = 293.15,																		# Initial temperature of the gas in Kelvin
	temp_old = 293.15,
	pressure_bar = 1.0,																	# Initial pressure of the gas in bar
	pressure_pa =  pressure_bar*1e5,													# Initial pressure of the gas in Pascal
	n_mol = pressure_pa * total_volume / (8.314*temp),				                    # Number of mol
	init_n_mol = copy(n_mol), 															# Initial number of mol
	real_n_particles = n_mol * 6.022e23/4,												# Real number of Particles in model: Reduction for simplicity
    n_particles = real_n_particles/1e23,												# Number of Particles in simulation model
	n_particles_old 		= copy(n_particles),
	molar_mass 				= 4.0,																# Helium Gas mass in atomic mass units
	mass_kg 				= molar_mass * 1.66053906660e-27,									# Convert atomic/molecular mass to kg
	mass_gas 				= round(n_mol * molar_mass, digits=3),								# Mass of gas
	radius 					= 8.0,																			# Radius of Particles in the model
	e_internal = 3/2 * n_mol * 8.314 * temp,											# Inner energy of the gas
	entropy_change = 0.0,
	old_scaled_speed 		= 0.0,																# Scaled speed of the previous step
	step 					= 0,																# Step counter
	max_speed 				= 8000.0,																# Change in entropy of the gas
	extent = (width,width),																	# Extent of Particles space
)
    space = ContinuousSpace(extent; spacing = 2.5)

	properties = Dict(
		:n_particles		=> n_particles,
		:temp				=> temp,
		:temp_old			=> temp_old,
		:total_volume		=> total_volume,
		:e_internal			=> e_internal,
		:old_scaled_speed => old_scaled_speed,
		:entropy_change 	=> entropy_change,
		:pressure_pa		=> pressure_pa,
		:pressure_bar		=> pressure_bar,
		:real_n_particles	=> real_n_particles,
		:n_mol		=> n_mol,
		:n_particles_old	=> n_particles_old,
		:volume	=> volume,
		#:temp_old	=> temp_old,
		:init_n_mol	=> init_n_mol,
		:gases		=> gases,
		:molar_mass		=> molar_mass,
		:mass_kg		=> mass_kg,
		:mass_gas	=> mass_gas,
		:step => 0,
		:cylinder_command => 0, 
		:cylinder_pos => 500,
		:reduce_volume_merker => 500,
		:modes				=> modes,
		:mode				=> mode,
		:objective 			=> create_heatmap(width),
		:radius				=> radius,
		:max_speed			=> max_speed,
	)


    model = ABM( Particle, space; properties, scheduler = Schedulers.Randomly())
	scaled_speed = calc_and_scale_speed(model)
	model.old_scaled_speed = scaled_speed
	for _ in 1:n_particles
		vel = Tuple( 2rand(2).-1)
		vel = vel ./ norm(vel)  # ALWAYS maintain normalised state of vel!
        add_agent!( model, vel, mass_kg, scaled_speed, radius, non_id, -Inf)
	end
    return model
end
#-----------------------------------------------------------------------------------------
"""
calc_total_vol_dimension( me, model)

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
	agent_step!( me, model)

This is the heart of the IdealGas model: It calculates how Particles collide with each other,
while conserving momentum and kinetic energy.
"""
function agent_step!(me::Particle, model::ABM)
	her = random_nearby_agent( me, model, 2*me.radius)	# Grab nearby particle
	if her !== nothing && her.id < me.id && her.id != me.prev_partner
        # New collision partner has not already been handled and is not my previous partner:
        me.prev_partner = her.id           # Update previous partners to avoid repetitive juddering collisions.
        her.prev_partner = me.id           # ditto for the other agent.

        # Compute relative position and velocity:
        rel_pos = me.pos .- her.pos
        rel_vel = me.vel .- her.vel

        # Compute collision impact vector:
        distance_sq = sum(rel_pos .^ 2)
        velocity_dot = sum(rel_vel .* rel_pos)
        impulse = 2 * me.mass * her.mass / (me.mass + her.mass) * velocity_dot ./ distance_sq .* rel_pos

        # Update velocities according to the impulse
		me.vel = (me.vel[1] - (impulse ./ me.mass)[1], me.vel[2] - (impulse ./ me.mass)[2])
		her.vel = (her.vel[1] + (impulse ./ her.mass)[1], her.vel[2] + (impulse ./ her.mass)[2])

        # Update speeds based on new velocities
        me.speed = norm(me.vel)
        her.speed = norm(her.vel)
    	end

	check_particle_near_border!(me, model)

	# Zylinder Steuerug 

	if model.cylinder_command == 1
		button_reduce_volume!(me, model)
		#println("zylinder fährt ein")
	elseif model.cylinder_command == 0
		model.reduce_volume_merker = model.cylinder_pos
	elseif model.cylinder_command == 2
		button_increase_volume!(me,model)
		model.reduce_volume_merker = model.space.extent[1] # solange Funtkion aktiv wird die Grenze bei check_particle_near_border aufgehoben 
	end 
	
	move_agent!(me, model, me.speed)
end
#----------------------------------------------------------------------------------------

function check_particle_near_border!(me, model)
    x, y = me.pos

    if x < 1.8 + me.radius/2 && model.step - me.last_bounce > 3
        me.vel = (-me.vel[1], me.vel[2])
        me.last_bounce = model.step
    elseif x > model.reduce_volume_merker - 1.8 && model.step - me.last_bounce > 3
        me.vel = (-me.vel[1], me.vel[2])
        me.last_bounce = model.step
    end
    if y < 1.8 + me.radius/2 && model.step - me.last_bounce > 3
        me.vel = (me.vel[1], -me.vel[2])
        me.last_bounce = model.step			
    elseif y > model.space.extent[2] - 1.8 && model.step - me.last_bounce > 3 
        me.vel = (me.vel[1], -me.vel[2])
        me.last_bounce = model.properties[:step]
    end
end

#-----------------------------------------------------------------------------------------
function button_increase_volume!(me, model)

	x,y = me.pos

	println(model.cylinder_pos)

	if model.cylinder_pos > 499.5 
    	println("zylinder ist in Ursprunngsposition")
	
	elseif x > model.cylinder_pos 
			me.vel = (-me.vel[1], me.vel[2])	
			me.speed = me.speed/2 # Anahme: die Hälfte der Energie wird abgegeben 
	end 
	
end

function button_reduce_volume!(me, model)
    x, y = me.pos

    # Überprüfen, ob y > 500 und falls ja, setzen Sie y auf 500 und invertieren Sie die y-Geschwindigkeit
	if model.cylinder_pos < 250 
    	#println("zylinder ist voll ausgefahren")
		model.reduce_volume_merker = model.cylinder_pos # neue Grenze
	else
		
		#println(model.cylinder_position)

     if x > model.cylinder_pos
        if model.properties[:step] - me.last_bounce < 3 # wenn der letzte Treffer noch nicht lange her war 
            me.speed = me.speed +1
			me.vel = (me.vel[1] - 0.5 , me.vel[2]) # x-Komponente wird durch Aufprall mit Wand verstärkt 
			me.vel = me.vel ./ norm(me.vel) 
			println("nächste Erhöhung ")
		 	println(me.speed)
			println(me.id)

        end
        if model.properties[:step] - me.last_bounce > 3 # wenn der letzte Treffer schon länger her war 
		 me.vel = (-me.vel[1], me.vel[2]) # change direction
		 me.vel = (me.vel[1] - 0.5 , me.vel[2]) # x-richtung erhöhen 
		 me.vel = me.vel ./ norm(me.vel) # Einheitsvektor erstellen 
         me.speed = me.speed + 1
		 println("Erhöhung ")
		 println(me.speed)
         me.last_bounce = model.properties[:step]
        end
     end
    end

end


#-----------------------------------------------------------------------------------------
"""
	model_step!( model)

	calculate the quantities, based on the chosen mode (Specifies which variables are constant)
	calculate the quantities, based on the chosen mode (Specifies which variables are constant)
"""
function model_step!(model::ABM)

	
	model.entropy_change = calc_entropy_change(model)
	model.e_internal = calc_internal_energy(model)

	scaled_speed = calc_and_scale_speed(model)
	# Set speed of all particles the same when approaching absolute zero (Downwards from T=60 K)
	if scaled_speed < 1
		for particle in allagents(model)
			particle.speed = scaled_speed
		end
	else # Else keep the difference in speed to the root mean square speed of the previous step (but scale it for visuals)
		for particle in allagents(model)
			particle.speed = scaled_speed + (particle.speed - model.old_scaled_speed) / (7/4)
		end
	end
	model.old_scaled_speed = scaled_speed

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
		function set_slider(value, slider, slider_value, unit)
			slider_value.text[] = string(round(value, digits=2), " ", unit)
			set_close_to!(slider, round(value, digits=2))
		end

		function add_or_remove_agents!(model)
			if model.n_particles_old < model.n_particles
				scaled_speed = calc_and_scale_speed(model)
				for _ in model.n_particles_old:model.n_particles
					vel = Tuple( 2rand(2).-1)
					vel = vel ./ norm(vel)  # ALWAYS maintain normalised state of vel!
					add_agent!( model, vel, model.mass_kg, scaled_speed, model.radius, non_id, -Inf)
				end
		
				model.n_particles_old = model.n_particles
		
			elseif model.n_particles_old > model.n_particles
		
				for _ in model.n_particles:model.n_particles_old
					agent = random_agent(model)
					kill_agent!(agent, model)
				end
		
				model.n_particles_old = model.n_particles
			end
		end

		function create_custom_slider(slider_space, row_num, labeltext, fontsize, range, unit, startvalue)
			label = Label(slider_space[row_num, 0], labeltext, fontsize=fontsize)
			slider = Slider(slider_space[row_num, 1], range=range, startvalue=startvalue)
			slider_value = Label(slider_space[row_num, 2], string(slider.value[]) * " " * unit)
			return label, slider, slider_value
		end

		model = idealgas()
		
		# Resolution of Makie Window based on monitor size 
		monitor = GLMakie.GLFW.GetPrimaryMonitor()
		width = GLMakie.MonitorProperties(monitor).videomode.width
		height = GLMakie.MonitorProperties(monitor).videomode.height
		
		plotkwargs = (;
    		ac = :skyblue3,
    		scatterkwargs = (strokewidth = 1.0,),
			as = 8.0,
			add_colorbar = false,
			colormap=:greys,
			colorrange=(0, 1),
			heatarray=:objective,
		)
	
		entropy(model) = model.entropy_change
		mdata = [entropy]
		mlabels = ["ΔS in [J/K] (Entropieänderung)"]
	
		playground,abmobs = abmplayground( model, idealgas;
			agent_step!,
			model_step!,
			mdata,
			mlabels,
			figure = (; resolution = (width*0.75, height*0.75)),
			ac =:skyblue3,
			plotkwargs...
		)

		# Figure Objekten neues Layout zuweisen durch feste Reihenfolge in figure.content[i]
		model_plot = playground.content[1]	# model 	
		playground[0:2,0] = model_plot
		entropy_plot = playground.content[7]
		playground[0:1,2][1,0:1] = entropy_plot

		# Sliders
		playground[2,1] = playground.content[2]
		slider_space = playground[2,2] = GridLayout()

		# Buttons to change volume 
		vol_change_btns = playground[1,1] = GridLayout()
		increase_vol_btn = Button(vol_change_btns[0,1:2], label = "Increase\nVolumen")# = print("increase"))#increase_vol_const())
		pause_vol_btn = Button(vol_change_btns[0,3], label = "Pause")
		decrease_vol_btn = Button(vol_change_btns[0,4:5], label = "Decrease\nVolumen")# = print("decrease"))#decrease_vol_const())

		on(increase_vol_btn.clicks) do _
			model.cylinder_command = 1
		end  

		on(pause_vol_btn.clicks) do _
			model.cylinder_command = 0
		end 

		on(decrease_vol_btn.clicks) do _
			model.cylinder_command = 2
		end

		# Buttons
		gl_buttons = playground[3,1] = GridLayout()
		gl_buttons[0,2] = playground.content[3]
		gl_buttons[0,3] = playground.content[4]
		gl_buttons[0,4] = playground.content[5]
		gl_buttons[0,5] = playground.content[6]

		# GridLayouts
		gl_sliders = playground[4,:] = GridLayout()
		gl_dropdowns = playground[3,0] = GridLayout()
		gl_labels = playground[0,1] = GridLayout()

		# Dropdowns
		gas_dropdown = Menu(gl_dropdowns[0,0], options = keys(model.gases), default = "Helium")
		mode_dropdown = Menu(gl_dropdowns[0,1], options = keys(model.modes), default = "Temperatur:Druck")

		# Labels
		#pressure_label = Label(gl_labels[2,0], "Druck: " * string(round(model.pressure_bar, digits=2))* " Bar", fontsize=22)
		mass_label = Label(gl_labels[0,0], "Masse: " * string(model.mass_gas)* " g", fontsize=22)
		#volume_label = Label(gl_labels[1,0], "Volumen: " * string(round(model.total_volume, digits=2))* " m³ ; " * string(round(model.total_volume * 1000, digits=2)) * " L", fontsize=22)
		e_internal_label = Label(gl_labels[1,0], "Eᵢ: " * string(round(model.e_internal, digits=2)) * " J", fontsize=22)


		# multipliers and quotients to set min, max and step of sliders based on current values
		slider_multiplier_min = 0.5
		slider_multiplier_max = 1.5
		slider_step_quotient = 100.0

		temp_slider_label, temp_slider, temp_slider_value = create_custom_slider(slider_space, 0, "Temperatur: ", 16, model.temp*slider_multiplier_min:model.temp/slider_step_quotient:model.temp*slider_multiplier_max, "K", model.temp)
		pressure_slider_bar_label, pressure_slider_bar, pressure_slider_bar_value = create_custom_slider(slider_space, 1, "Druck[Bar]: ", 16, model.pressure_bar*slider_multiplier_min:model.pressure_bar/slider_step_quotient:model.pressure_bar*slider_multiplier_max, "Bar", model.pressure_bar)
		pressure_slider_pa_label, pressure_slider_pa, pressure_slider_pa_value = create_custom_slider(slider_space, 2, "Druck[Pa]: ", 16, model.pressure_pa*slider_multiplier_min:model.pressure_pa/slider_step_quotient:model.pressure_pa*slider_multiplier_max, "Pa", model.pressure_pa)
		volume_slider_label, volume_slider, volume_slider_value = create_custom_slider(slider_space, 3, "Volumen: ", 16, model.total_volume*slider_multiplier_min:model.total_volume/slider_step_quotient:model.total_volume*slider_multiplier_max, "m³", model.total_volume)
		n_mol_slider_label, n_mol_slider, n_mol_slider_value = create_custom_slider(slider_space, 4, "Stoffmenge: ", 16, model.n_mol*slider_multiplier_min:model.n_mol/slider_step_quotient:model.n_mol*slider_multiplier_max, "mol", model.n_mol)

		is_updating = false

		on(abmobs.model) do _
			e_internal_label.text[] = string("Eᵢ: ", string(round(model.e_internal)), " J")
			#pressure_label.text[] = string("Druck: ", string(round(model.pressure_bar, digits=2)), " Bar")
			if model.mass_gas > 999.9
				mass_label.text[] = string("Masse: ", string(round(model.mass_gas/1000, digits=3), " kg"))
			else
				mass_label.text[] = string("Masse: ", string(model.mass_gas), " g")
			end
			#volume_label.text[] = string("Volumen: " * string(round(model.total_volume, digits=2))* " m³ ; " * string(round(model.total_volume * 1000, digits=2)) * " L")

		end

		on(gas_dropdown.selection) do selected_gas
			new_molarmass = model.gases[selected_gas]
			model.molar_mass = new_molar_mass
			model.mass_kg = new_molar_mass * 1.66054e-27
			model.mass_gas = round(model.n_mol * model.molar_mass, digits=3)
			
		end

		on(mode_dropdown.selection) do selected_mode
			model.mode = model.modes[selected_mode]
		end

		# on change of temperature slider
		on(temp_slider.value) do temp
			if model.mode == "temp-druck" || model.mode == "temp-vol"
				model.temp = temp[]
				temp_slider_value.text[] = string(round(temp[], digits=2)) * " K"
			end
			
			# If the mode is "temp-druck" the pressure is calculated
			if model.mode == "temp-druck"
				model.pressure_pa = TD_Physics.calc_pressure(model)
				model.pressure_bar = model.pressure_pa / 1e5
				set_slider(model.pressure_pa, pressure_slider_pa, pressure_slider_pa_value, "Pa")
				set_slider(model.pressure_bar, pressure_slider_bar, pressure_slider_bar_value, "Bar")

				# set the range of the sliders
				pressure_slider_pa.range = model.pressure_pa * slider_multiplier_min:model.pressure_pa/slider_step_quotient:model.pressure_pa * slider_multiplier_max
				pressure_slider_bar.range = model.pressure_bar * slider_multiplier_min:model.pressure_bar / slider_step_quotient:model.pressure_bar * slider_multiplier_max

			# If the mode is "temp-vol" the volume is calculated
			elseif model.mode == "temp-vol"
				model.total_volume = calc_volume(model)
				set_slider(model.total_volume, volume_slider, volume_slider_value, "m³")
				model.volume = calc_total_vol_dimension(model.total_volume)

				# set the range of the sliders
				volume_slider.range = model.total_volume * slider_multiplier_min:model.total_volume/slider_step_quotient:model.total_volume * slider_multiplier_max
			end
		end

		# on change of pressure bar slider
		on(pressure_slider_bar.value) do pressure
			if is_updating
				return
			end
			is_updating = true
			if model.mode == "druck-vol" || model.mode == "druck-temp"
				model.pressure_bar = pressure[]
				model.pressure_pa = pressure[] * 1e5
				pressure_slider_bar_value.text[] = string(round(model.pressure_bar, digits=2)) * " Bar"
				set_slider(model.pressure_pa , pressure_slider_pa, pressure_slider_pa_value, "Pa")
				
				# If the mode is "druck-vol" the volume is calculated
				if model.mode == "druck-vol"
					model.total_volume = calc_volume(model)
					set_slider(model.total_volume, volume_slider, volume_slider_value, "m³")
					model.volume = calc_total_vol_dimension(model.total_volume)

					# set the range of the sliders
					volume_slider.range = model.total_volume * slider_multiplier_min:model.total_volume/slider_step_quotient:model.total_volume * slider_multiplier_max
				# If the mode is "druck-temp" the temperature is calculated
				elseif model.mode == "druck-temp"
					model.temp = calc_temperature(model)
					set_slider(model.temp, temp_slider, temp_slider_value, "K")

					# set the range of the sliders
					temp_slider.range = model.temp * slider_multiplier_min:model.temp/slider_step_quotient:model.temp * slider_multiplier_max

				end
			end
			is_updating = false
		end

		# on change of pressure pa slider
		on(pressure_slider_pa.value) do pressure
			if is_updating
				return
			end
			is_updating = true
			if model.mode == "druck-vol" || model.mode == "druck-temp"
				model.pressure_pa = pressure[]
				model.pressure_bar = pressure[] / 1e5
				pressure_slider_pa_value.text[] = string(round(model.pressure_pa, digits=2)) * " Pa"
				set_slider(model.pressure_bar, pressure_slider_bar, pressure_slider_bar_value, "Bar")

				# If the mode is "druck-vol" the volume is calculated
				if model.mode == "druck-vol"
					model.total_volume = calc_volume(model)
					set_slider(model.total_volume, volume_slider, volume_slider_value, "m³")
					model.volume = calc_total_vol_dimension(model.total_volume)

					# set the range of the sliders
					volume_slider.range = model.total_volume * slider_multiplier_min:model.total_volume/slider_step_quotient:model.total_volume * slider_multiplier_max
				# If the mode is "druck-temp" the temperature is calculated
				elseif model.mode == "druck-temp"
					model.temp = calc_temperature(model)
					set_slider(model.temp, temp_slider, temp_slider_value, "K")

					# set the range of the sliders
					temp_slider.range = model.temp * slider_multiplier_min:model.temp/slider_step_quotient:model.temp * slider_multiplier_max

				end
			end
			is_updating = false
		end

		# on change of volume slider
		on(volume_slider.value) do volume
			if model.mode == "vol-druck" || model.mode == "vol-temp"
				model.total_volume = volume[]
				volume_slider_value.text[] = string(round(volume[], digits=2)) * " m³"
				model.volume = calc_total_vol_dimension(model.total_volume)

				# If the mode is "vol-druck" the pressure is calculated
				if model.mode == "vol-druck"
					model.pressure_pa = calc_pressure(model)
					model.pressure_bar = model.pressure_pa / 1e5
					set_slider(pressure, pressure_slider_pa, pressure_slider_pa_value, "Pa")
					set_slider(pressure / 1e5, pressure_slider_bar, pressure_slider_bar_value, "Bar")

					# set the range of the sliders
					pressure_slider_pa.range = model.pressure_pa * slider_multiplier_min:model.pressure_pa/slider_step_quotient:model.pressure_pa * slider_multiplier_max
					pressure_slider_bar.range = model.pressure_bar * slider_multiplier_min:model.pressure_bar / slider_step_quotient:model.pressure_bar * slider_multiplier_max

				# If the mode is "vol-temp" the temperature is calculated
				elseif model.mode == "vol-temp"
					model.temp = calc_temperature(model)
					set_slider(model.temp, temp_slider, temp_slider_value, "K")

					# set the range of the sliders
					temp_slider.range = model.temp * slider_multiplier_min:model.temp/slider_step_quotient:model.temp * slider_multiplier_max

				end
			end
		end

		# on change of n_mol slider
		on(n_mol_slider.value) do n_mol
			if model.mode == "mol-temp" || model.mode == "mol-druck"
				model.n_mol = n_mol[]
				n_mol_slider_value.text[] = string(round(n_mol[], digits=2)) * " mol"
				model.n_particles = model.n_mol * 6.022e23 / 1e22 / 8
				add_or_remove_agents!(model)				

				# If the mode is "n_mol-temp" the temperature is calculated
				if model.mode == "mol-temp"
					model.temp = calc_temperature(model)
					set_slider(model.temp, temp_slider, temp_slider_value, "K")

					# set the range of the sliders
					temp_slider.range = model.temp * slider_multiplier_min:model.temp/slider_step_quotient:model.temp * slider_multiplier_max


				# If the mode is "n_mol-druck" the pressure is calculated
				elseif model.mode == "mol-druck"
					model.pressure_pa = calc_pressure(model)
					model.pressure_bar = model.pressure_pa / 1e5
					set_slider(model.pressure_pa, pressure_slider_pa, pressure_slider_pa_value, "Pa")
					set_slider(model.pressure_bar, pressure_slider_bar, pressure_slider_bar_value, "Bar")

					# set the range of the sliders
					pressure_slider_pa.range = model.pressure_pa * slider_multiplier_min:model.pressure_pa/slider_step_quotient:model.pressure_pa * slider_multiplier_max
					pressure_slider_bar.range = model.pressure_bar * slider_multiplier_min:model.pressure_bar / slider_step_quotient:model.pressure_bar * slider_multiplier_max
				end
			end
		end
		playground
	end
end	# of module IdealGas