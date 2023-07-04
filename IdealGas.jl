#========================================================================================#
"""
	IdealGas
änderung
Extending SimpleParticles to conserve kinetic energy and momentum.

Author: Francisco Hella, Felix Rollbühler, Melanie Heinrich, Jan Wichmann, 22/06/23
"""
module IdealGas

include("AgentTools.jl")
include("PhysicalModel.jl")
include("ViewModel.jl")
using Agents, LinearAlgebra, GLMakie, InteractiveDynamics, .AgentTools, .PhysicalModel

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
	last_bounce::Float64			# Time of last collision
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
	gases = Dict("Helium" => 4.0, "Hydrogen" => 1.0, "Oxygen" => 32.0),	# Gas types
	modes = Dict("Temperatur:Druck" => "temp-druck", 					# modes of calculations
				#"Temperatur:Volumen" => "temp-vol",
				 "Druck:Temperatur" => "druck-temp",
				 #"Druck:Volumen" => "druck-vol",
				 "Volumen:Temperatur" => "vol-temp",
				 "Volumen:Druck" => "vol-druck",										
				"Mol : Temperatur" => "mol-temp",
				"Mol : Druck" => "mol-druck",
				),				
	mode = "temp-druck",
	total_volume_m3 = 0.25,												# Total Volumen in m^3
	width = 500,														# Widht and height of the container in cm
	start_volume = copy(total_volume_m3),								# Initial volume of the container
	volume = calc_total_vol_dimension,									# Dimension of the container
	temp = 293.15,														# Initial temperature of the gas in Kelvin
	temp_old = 293.15,													# Temperature of the gas in Kelvin
	pressure_bar = 1.0,													# Initial pressure of the gas in bar
	pressure_pa =  100000,												# Initial pressure of the gas in Pascal
	n_mol = pressure_pa * total_volume_m3 / (8.314*temp),				# Number of mol
	init_n_mol = copy(n_mol), 											# Initial number of mol
	real_n_particles = n_mol * 6.022e23,								# Real number of Particles in model: Reduction for simplicity
    n_particles = round(real_n_particles/1e23, digits=0),				# Number of Particles in simulation model
	n_particles_old 		= copy(n_particles), 						# Number of Particles in simulation model
	molar_mass 				= 4.0,										# Helium Gas mass in atomic mass units
	mass_kg 				= molar_mass * 1.66053906660e-27,			# Convert atomic/molecular mass to kg
	mass_gas 				= round(n_mol * molar_mass, digits=3),		# Mass of gas
	radius 					= 12.0,										# Radius of Particles in the model
	e_internal = 3/2 * n_mol * 8.314 * temp,							# Inner energy of the gas
	entropy_change = 0.0,												# Change in entropy of the gas
	old_scaled_speed 		= 0.0,										# Scaled speed of the previous step
	step 					= 0,										# Step counter
	max_speed 				= 8000.0,									# Change in entropy of the gas
	extent = (width,width),												# Extent of Particles space
)
    space = ContinuousSpace(extent; spacing = 2.5)

	properties = Dict(
		:n_particles		=> n_particles,
		:temp				=> temp,
		:temp_old			=> temp_old,
		:total_volume_m3	=> total_volume_m3,
		:e_internal			=> e_internal,
		:old_scaled_speed 	=> old_scaled_speed,
		:entropy_change 	=> entropy_change,
		:pressure_pa		=> pressure_pa,
		:pressure_bar		=> pressure_bar,
		:real_n_particles	=> real_n_particles,
		:n_mol				=> n_mol,
		:n_particles_old	=> n_particles_old,
		:volume				=> volume,
		:init_n_mol			=> init_n_mol,
		:gases				=> gases,
		:molar_mass			=> molar_mass,
		:mass_kg			=> mass_kg,
		:mass_gas			=> mass_gas,
		:step 				=> 0,
		:cylinder_command 	=> 0, 
		:cylinder_pos 		=> width,
		:reduce_volume_merker => width,
		:modes				=> modes,
		:mode				=> mode,
		:heatmap 			=> ones(width, width),
		:radius				=> radius,
		:max_speed			=> max_speed,
		:width				=> width,
		:start_volume		=> start_volume,
	)


    model = ABM( Particle, space; properties, scheduler = Schedulers.Randomly()) #create model
	scaled_speed = calc_and_scale_speed(model) #calculate and scale speed
	model.old_scaled_speed = scaled_speed # set old scaled speed
	for _ in 1:n_particles #add particles
		vel = Tuple( 2rand(2).-1) #random velocity
		vel = vel ./ norm(vel)  # normalize velocity
        add_agent!( model, vel, mass_kg, scaled_speed, radius, non_id, -Inf) #add particle
	end
    return model
end
#-----------------------------------------------------------------------------------------
"""
calc_total_vol_dimension( me, model)

Calculates volume/dimension of a 3D-Space with [x, y = 500, z=1], based on a given value of total volume.
"""
function calc_total_vol_dimension(volume_m3, y_axis=500.0)
	x_axis = volume_m3/(y_axis/1000)
	println(x_axis)
 	return [x_axis, y_axis, 1.0] # return volume/dimension
end

#-----------------------------------------------------------------------------------------
"""
    change_heatmap!(model::ABM)

		Changes the underlying heatmap as a representation of the chanching gas-tank.
"""
	function change_heatmap!(model::ABM)
		line = fill(1.0, model.width)

		for e in 1:model.width
			e = round(e)
			if e > model.cylinder_pos
				line[e] = 0.0
			end
		end
		for i in 1:model.width
			model.heatmap[:,i] = line
		end 


		#for i in CartesianIndices(model.heatmap) # Iterate over all indices of the heatmap
		#	if i[1] >= round(model.cylinder_pos) # If the index is in the cylinder
		#		model.heatmap[i] = 0.0 			 # Set the value to 0
		#	end
		#	if i[1] <= round(model.cylinder_pos) # If the index is in the cylinder
		#		model.heatmap[i] = 1.0 			 # Set the value to 0
		#	end
		#end
	end
#-----------------------------------------------------------------------------------------
"""
	agent_step!( me, model)

This is the heart of the IdealGas model: It calculates how Particles collide with each other,
while conserving momentum and kinetic energy.
"""
function agent_step!(me::Particle, model::ABM)
	her = random_nearby_agent( me, model, 2*me.radius)	# Grab nearby particle
	if her !== nothing && her.id != me.prev_partner
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
		me.vel = (me.vel[1] - (impulse ./ me.mass)[1], me.vel[2] - (impulse ./ me.mass)[2]) #
		her.vel = (her.vel[1] + (impulse ./ her.mass)[1], her.vel[2] + (impulse ./ her.mass)[2])

        # Update speeds based on new velocities
        me.speed = norm(me.vel)
        her.speed = norm(her.vel)
    	end

	check_particle_near_border!(me, model)

	# Zylinder control 
	if model.cylinder_command == 1
		button_reduce_volume!(me, model)
	elseif model.cylinder_command == 0
		model.reduce_volume_merker = model.cylinder_pos
	elseif model.cylinder_command == 2
		button_increase_volume!(me,model)
		model.reduce_volume_merker = model.space.extent[1] # as long as the function is active, the border of check_particle_near_border! is deacrivated
	end 
	move_agent!(me, model, me.speed)
end
#----------------------------------------------------------------------------------------

function check_particle_near_border!(me, model)
    x, y = me.pos # Get current position

    if x < 1.8 + me.radius/2 && model.step - me.last_bounce > 3 # Check if near left border
        me.vel = (-me.vel[1], me.vel[2]) # Reflect x-velocity
        me.last_bounce = model.step # Update last bounce
    elseif x > model.reduce_volume_merker - 1.8 && model.step - me.last_bounce > 3 # Check if near right border
        me.vel = (-me.vel[1], me.vel[2]) # Reflect x-velocity
        me.last_bounce = model.step # Update last bounce
    end
    if y < 1.8 + me.radius/2 && model.step - me.last_bounce > 3 # Check if near bottom border
        me.vel = (me.vel[1], -me.vel[2]) # Reflect y-velocity
        me.last_bounce = model.step # Update last bounce
    elseif y > model.space.extent[2] - 1.8 && model.step - me.last_bounce > 3 # Check if near top border 
        me.vel = (me.vel[1], -me.vel[2]) # Reflect y-velocity
        me.last_bounce = model.properties[:step] # Update last bounce
    end
end

#-----------------------------------------------------------------------------------------
function button_increase_volume!(me, model)
	x,y = me.pos

	if model.cylinder_pos > 499.5 
	
	elseif x > model.cylinder_pos 
		me.vel = (-me.vel[1], me.vel[2])	
		me.speed = me.speed/2 # Assumption: the particle loses half of its speed when hitting the piston
	end 
end
#-----------------------------------------------------------------------------------------
function button_reduce_volume!(me, model)
    x, y = me.pos

    #  Check if y > 500 and if so set y to 500 and invert the y velocity
	if model.cylinder_pos < 250 
		model.reduce_volume_merker = model.cylinder_pos # new border
	else
     	if x > model.cylinder_pos
        	if model.properties[:step] - me.last_bounce < 3 # if the last bounce was recent
            	me.speed = me.speed +1
				me.vel = (me.vel[1] - 0.5 , me.vel[2]) # x-component is increased when hitting the piston
				me.vel = me.vel ./ norm(me.vel) 
        	end
			if model.properties[:step] - me.last_bounce > 3 # if the last bounce was a while ago
				me.vel = (-me.vel[1], me.vel[2]) # change direction
				me.vel = (me.vel[1] - 0.5 , me.vel[2]) # increase x-component
				me.vel = me.vel ./ norm(me.vel) # create unit vector
				me.speed = me.speed + 1
				me.last_bounce = model.properties[:step]
			end
    	end
    end
end


#-----------------------------------------------------------------------------------------
function model_step!(model::ABM)
	"""

	This function is called at every step of the simulation.
	In this case, it is used to calculate the entropy change and the internal energy of the system.
	It also updates the speed of the particles and moves the piston

	"""
	
	model.entropy_change = calc_entropy_change(model) # calculate the entropy change
	model.e_internal = calc_internal_energy(model) # calculate the internal energy

	scaled_speed = calc_and_scale_speed(model) # calculate the root mean square speed and scale it for visuals

	# Set speed of all particles the same when approaching absolute zero (Downwards from T=60 K)
	if scaled_speed < 1 # If the speed is below 1, set it to 1 for all particles
		for particle in allagents(model) # for all particles in the model
			particle.speed = scaled_speed # set the speed to the scaled value
		end
	else # Else keep the difference in speed to the root mean square speed of the previous step (but scale it for visuals)
		for particle in allagents(model) # for all particles in the model
			particle.speed = scaled_speed + (particle.speed - model.old_scaled_speed) / (7/4) # set the speed to the scaled value
		end
	end
	model.old_scaled_speed = scaled_speed # set the old scaled speed to the current scaled speed

	model.step += 1.0

	if model.cylinder_command == 1 && model.cylinder_pos > 250 # expand cylinder
		model.cylinder_pos = model.cylinder_pos - 0.3 # Change cylinder position
		if  mod(model.step, 3) == 0 # update heatmap every 3 steps
			change_heatmap!(model) # update heatmap
		end
	elseif model.cylinder_command == 2 && model.cylinder_pos < 500 # compress cylinder
		model.cylinder_pos = model.cylinder_pos + 0.3 # Change cylinder position
		if  mod(model.step, 3) == 0 # update heatmap every 3 steps
			change_heatmap!(model) # update heatmap
		end
	end

	model.total_volume_m3 = (model.cylinder_pos / model.width) * model.start_volume # Volumen wird berechnet

	if model.mode == "vol-temp" # If the mode is volume-temperature
		model.temp = PhysicalModel.calc_temperature(model) # Calculate the temperature
	else
		model.pressure_pa = PhysicalModel.calc_pressure(model) # Calculate the pressure
		model.pressure_bar = model.pressure_pa / 1e5 # Calculate the pressure in bar
	end
end

#----------------------------------------------------------------------------------------

"""
	demo()

Run a simulation of the IdealGas model and init the UI.
"""
	function demo()

		model = idealgas()
		
		monitor = GLMakie.GLFW.GetPrimaryMonitor() # get primary monitor
		width = GLMakie.MonitorProperties(monitor).videomode.width # get width of monitor
		height = GLMakie.MonitorProperties(monitor).videomode.height # get height of monitor

		heatkwargs = (colorrange = (0,1), colormap =:greys,)
		
		plotkwargs = (; # kwargs for the plot
    		ac = :skyblue3, # color of the particles
    		scatterkwargs = (strokewidth = 1.0,), # kwargs for the scatterplot
			as = 23.0, # size of the particles
			add_colorbar = false,
			heatarray=:heatmap, # type of heatmap
			framerate = 60, # Refreshrate of the simulation
			heatkwargs
		)
	
		entropy(model) = model.entropy_change # Entropieänderung als Funktion für die Visualisierung
		mdata = [entropy]
		mlabels = ["ΔS in [J/K] (Entropieänderung)"]
	
		playground,abmobs = abmplayground( model, idealgas; # create playground
			agent_step!,
			model_step!,
			mdata,
			mlabels,
			figure = (; resolution = (width*0.75, height*0.75)), # set resolution of the figure
			plotkwargs...
		)

		# Playground
		model_plot = playground.content[1] # plot for the visualization of the model
		model_plot.title = "Gasflasche\n(500x500x1)"
		model_plot.xlabel = "in cm"
		model_plot.ylabel = "in cm"
		playground[0:2,0] = model_plot 
		entropy_plot = playground.content[7] # plot for the graph of the entropy
		playground[0:1,1:2] = entropy_plot

		# GridLayouts
		#gl_sliders = playground[4,:] = GridLayout()
		vol_change_btns = playground[3,0] = GridLayout() # GridLayout for the volume change buttons
		gl_dropdowns = playground[5,0] = GridLayout() # GridLayout for the dropdowns
		gl_labels = playground[2,1:2] = GridLayout() # GridLayout for the labels
		gl_sliders = playground[3:4,1:2] = GridLayout() # GridLayout for the sliders
		gl_abm_sliders = playground[5,1:2] = GridLayout() # GridLayout for the sliders of the ABM

		# Arrangement of the labels of the abm
		gl_buttons = playground[4,0] = GridLayout()
		gl_buttons[0,2:3] = playground.content[3] # step model button
		gl_buttons[0,0:1] = playground.content[4] # run model button
		gl_buttons[0,4:5] = playground.content[5] # reset model button
		gl_buttons[0,6:7] = playground.content[6] # clear data button

		# creation of the buttons
		increase_vol_btn = Button(vol_change_btns[0,3:4], label = "Volumen\nerhöhen", fontsize = 18, padding = (8, 8, 8, 8)) # button to increase the volume
		pause_vol_btn = Button(vol_change_btns[0,2], label = "Zylinder\nstoppen", fontsize = 18, padding = (8, 8, 8, 8)) # button to pause the volume change
		decrease_vol_btn = Button(vol_change_btns[0,0:1], label = "Volumen\nverringern", fontsize = 18, padding = (8, 8, 8, 8)) # button to decrease the volume

		# creation of the sliders
		gas_dropdown = Menu(gl_dropdowns[0,1], options = keys(model.gases), default = "Helium", fontsize=22) # dropdown for the selection of the gas
		mode_dropdown = Menu(gl_dropdowns[1,1], options = keys(model.modes), default = "Temperatur:Druck", fontsize=22) # dropdown for the selection of the mode

		# creation of the labels
		gas_label = Label(gl_dropdowns[0,0], "Gas: ", fontsize=22)# label for the gas
		mode_label = Label(gl_dropdowns[1,0], "Modus: ", fontsize=22) # label for the mode
		mass_label = Label(gl_labels[0,0:2], "Masse: " * string(model.mass_gas)* " g", fontsize=22) # label for the mass of the gas
		volume_label = Label(gl_labels[1,0:2], "Volumen: " * string(round(model.total_volume_m3, digits=4))* " m³ ; " * string(round(model.total_volume_m3 * 1000, digits=4)) * " L", fontsize=22) # label for the volume of the box
		e_internal_label = Label(gl_labels[2,0:2], "Eᵢ: " * string(round(model.e_internal, digits=2)) * " J", fontsize=22) # label for the internal energy of the gas
		warning_label = Label(gl_buttons[1,0:7], "'Reset-Button' nicht benutzen !!!", fontsize=22) # label for the warning of the reset button, because it is not working properly
		

		# Sliders
		gl_abm_sliders[0,0:2] = playground.content[2]# Build-in Slider von abmexploration

		temp_slider_label, temp_slider, temp_slider_value = ViewModel.create_custom_slider(gl_sliders, 0, "Temperatur: ", 22, 0.0:0.01:1000.0, "K", model.temp) # slider for the temperature
		pressure_slider_bar_label, pressure_slider_bar, pressure_slider_bar_value = ViewModel.create_custom_slider(gl_sliders, 1, "Druck[Bar]: ", 22, 0.0:0.01:5.0, "Bar", model.pressure_bar) # slider for the pressure in bar
		pressure_slider_pa_label, pressure_slider_pa, pressure_slider_pa_value = ViewModel.create_custom_slider(gl_sliders, 2, "Druck[Pa]: ", 22, 0:1:500000, "Pa", model.pressure_pa) # slider for the pressure in Pa
		n_mol_slider_label, n_mol_slider, n_mol_slider_value = ViewModel.create_custom_slider(gl_sliders, 3, "Stoffmenge: ", 22, 0.0:0.01:20, "mol", model.n_mol) # slider for the amount of substance

		is_updating = false # variable to check if the model is updating

		on(abmobs.model) do _ # if the model is updated
			e_internal_label.text[] = string("Eᵢ: ", string(round(model.e_internal)), " J") # set the label for the internal energy of the gas
			if model.mass_gas > 999.9 # if the mass of the gas is greater than 999.9
				mass_label.text[] = string("Masse: ", string(round(model.mass_gas/1000, digits=3), " kg")) # set the label for the mass of the gas in kg
			else
				mass_label.text[] = string("Masse: ", string(model.mass_gas), " g") # set the label for the mass of the gas in g
			end
			volume_label.text[] = string("Volumen: " * string(round(model.total_volume_m3, digits=4))* " m³ ; " * string(round(model.total_volume_m3 * 1000, digits=4)) * " L") # set the label for the volume of the box

			ViewModel.set_slider(model.n_mol, n_mol_slider, n_mol_slider_value, "mol") # set the slider for the amount of substance
			ViewModel.set_slider(model.temp, temp_slider, temp_slider_value, "K") # set the slider for the temperature
			ViewModel.set_slider(model.pressure_pa, pressure_slider_pa, pressure_slider_pa_value, "Pa") # set the slider for the pressure in Pa
			ViewModel.set_slider(model.pressure_bar, pressure_slider_bar, pressure_slider_bar_value, "Bar") # set the slider for the pressure in bar
		end

		on(gas_dropdown.selection) do selected_gas # if the gas changes
			model.molar_mass = model.gases[selected_gas] # set the molar mass of the gas
			model.mass_kg = model.molar_mass * 1.66054e-27 # set the mass in kg of one molecule of the gas
			model.mass_gas = round(model.n_mol * model.molar_mass, digits=3) # set the mass of the gas
			
		end

		on(mode_dropdown.selection) do selected_mode # if the mode changes
			model.mode = model.modes[selected_mode] # set the mode
		end
		
		on(temp_slider.value) do temp # if the value of the temperature slider changes
			if model.mode == "temp-druck" || model.mode == "temp-vol" # if the mode is "temp-druck" or "temp-vol"
				model.temp = temp[] # set the temperature of the model to the value of the slider
				temp_slider_value.text[] = string(round(temp[], digits=2)) * " K" # set the text of the slider label
			end
			
			if model.mode == "temp-druck" # if the mode is "temp-druck"
				model.pressure_pa = PhysicalModel.calc_pressure(model) # calculate the pressure in Pa
				model.pressure_bar = model.pressure_pa / 1e5 # calculate the pressure in Bar
			end

			"""
			elseif model.mode == "temp-vol"
				model.total_volume_m3 = calc_volume(model)
				model.volume = calc_total_vol_dimension(model.total_volume_m3)

				# set the range of the sliders
				#volume_slider.range = model.total_volume_m3 * slider_multiplier_min:model.total_volume_m3/slider_step_quotient:model.total_volume_m3 * slider_multiplier_max
			end
			"""
		end
		
		on(pressure_slider_bar.value) do pressure # if the slider for the pressure in Bar is changing
			if is_updating # if the slider is updating
				return # end the function
			end
			is_updating = true
			if model.mode == "druck-vol" || model.mode == "druck-temp" # if the mode is "druck-vol" or "druck-temp"
				model.pressure_bar = pressure[] # set the pressure in Bar
				model.pressure_pa = pressure[] * 1e5 # set the pressure in Pa
				pressure_slider_bar_value.text[] = string(round(model.pressure_bar, digits=2)) * " Bar" # set the label of the slider
				
				if model.mode == "druck-vol" # if the mode is "druck-vol"
					model.total_volume_m3 = PhysicalModel.calc_volume(model) # calculate the total volume
					model.volume = calc_total_vol_dimension(model.total_volume_m3) # calculate the volume dimension

				elseif model.mode == "druck-temp" # If the mode is "druck-temp"
					model.temp = PhysicalModel.calc_temperature(model) # Calculate the temperature
				end
			end
			is_updating = false
		end

		on(pressure_slider_pa.value) do pressure # if the pressure slider in Pa changes
			if is_updating # if a slider is updating
				return # end the function
			end
			is_updating = true
			if model.mode == "druck-vol" || model.mode == "druck-temp" # if the mode is "druck-vol" or "druck-temp"
				model.pressure_pa = pressure[] # set the pressure in Pa
				model.pressure_bar = pressure[] / 1e5 # set the pressure in bar
				pressure_slider_pa_value.text[] = string(round(model.pressure_pa, digits=2)) * " Pa" # set the label of the slider

				if model.mode == "druck-vol" # if the mode is "druck-vol"
					model.total_volume_m3 = PhysicalModel.calc_volume(model) # calculate the volume
					model.volume = calc_total_vol_dimension(model.total_volume_m3) # calculate the dimension of the volume

				elseif model.mode == "druck-temp" # if the mode is "druck-temp"
					model.temp = PhysicalModel.calc_temperature(model) # calculate the temperature

				end
			end
			is_updating = false
		end

		on(n_mol_slider.value) do n_mol # if the slider for the number of moles changes
			if model.mode == "mol-temp" || model.mode == "mol-druck" # if the mode is "mol-temp" or "mol-druck"
				model.n_mol = n_mol[] # set the number of moles
				n_mol_slider_value.text[] = string(round(n_mol[], digits=2)) * " mol" # set the label of the slider
				model.real_n_particles = round(model.n_mol * 6.022e23, digits=0) # calculate the real number of particles
				model.n_particles = round(model.real_n_particles / 1e23, digits=0) # scale the number of particles to a reasonable/displayable number
				ViewModel.add_or_remove_agents!(model) # add or remove agents to match the scaled number of particles

				if model.mode == "mol-temp" # if the mode is "mol-temp"
					model.temp = PhysicalModel.calc_temperature(model) # calculate

				elseif model.mode == "mol-druck" # if the mode is "mol-druck"
					model.pressure_pa = PhysicalModel.calc_pressure(model) # calculate the pressure
					model.pressure_bar = model.pressure_pa / 1e5 # calculate the pressure in bar

				end
			end
		end

		on(increase_vol_btn.clicks) do _ # if the button to increase the volume is clicked
			model.cylinder_command = 2 # setze den Befehl zum Erhöhen des Volumens
		end  

		on(pause_vol_btn.clicks) do _ # if the button to pause the volume is clicked
			model.cylinder_command = 0 # set the command to pause the volume
		end 

		on(decrease_vol_btn.clicks) do _ # if the button to decrease the volume is clicked
			model.cylinder_command = 1 # set the command to decrease the volume
		end

		playground # draw the playground
	end
end	# end of module IdealGas