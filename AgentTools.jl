#========================================================================================#
"""
	AgentTools

This module provides a collection of utility functions for agent-based models that extendi
the Agents package towards conformity with NetLogo.

Authors:  Emilio Borelli, Nick Diercksen, Stefan Hausner, Niall Palfreyman, Dominik Pfister
Date: July 2022
"""
module AgentTools

using Agents, GLMakie, InteractiveDynamics, LinearAlgebra, Observables

import InteractiveUtils:@which

export abmplayground, multicoloured, dejong2, diffuse4, diffuse4!, diffuse8, diffuse8!,
		gradient, size, spectrum, turn!, valleys, wedge

#-----------------------------------------------------------------------------------------
# Module methods:
#-----------------------------------------------------------------------------------------
"""
	size( cs::ContinuousSpace)

Extend Base.size to ContinuousSpaces to allow heatmaps to work in abmplot.
"""
Base.size(cs::Agents.ContinuousSpace) = cs.dims

#-----------------------------------------------------------------------------------------
"""
	multicoloured(agent)

Return different colours depending on the id of the agent.
"""
const spectrum = [:black,:darkblue,:blue,:green,:violet,:crimson,:red,:orange,:yellow]
function multicoloured( agent::AbstractAgent)
	spectrum[1+agent.id%length(spectrum)]
end

#-----------------------------------------------------------------------------------------
"""
	wedge(p)

Draw a wedge-shaped marker pointing in the agent's facing (vel) direction.
"""
function wedge( agent::AbstractAgent; as=1)
	scale(
		rotate2D(
			Polygon(Point2f[(1, 0), (-0.5, 0.5), (-0.5, -0.5)]),
			atan(agent.vel[2], agent.vel[1])
		),
		as
	)
end

#-----------------------------------------------------------------------------------------
"""
	turn!( agent::AbstractAgent, angle)
	
Rotates agent's facing direction (vel).
"""
function turn!(agent::AbstractAgent, θ=pi/2)
	cosθ = cos(θ); sinθ = sin(θ)
	agent.vel = Tuple([cosθ -sinθ;sinθ cosθ]*collect(agent.vel))
end

#-----------------------------------------------------------------------------------------
"""
	gradient( pos, flow::AbstractArray, model::ABM)
	
Return the gradient vector for the given flow at this position in the model.
"""
function gradient( pos, flow::AbstractArray, model::ABM; h::Float64=1.0)
	nbhdflow = map([(-h,0),(h,0),(0,-h),(0,h)]) do step
		flow[ get_spatial_index( normalize_position(pos.+step,model), flow, model) ]
	end

	((nbhdflow[2]-nbhdflow[1]), (nbhdflow[4]-nbhdflow[3])) ./ 2h
end

#-----------------------------------------------------------------------------------------
"""
	valleys( width)

Return a 2D map of a multimodal valleys landscape with dimensions (width, width).
"""
function valleys( width::Int)
	xs = repeat( range(-3,3,width), 1, width)

	map(
		(x,y) -> (1/3)*exp(-((x + 1)^2) - (y^2)) +
			10*(x/5 - (x^3) - (y^5)) * exp(-(x^2) - (y^2)) -
			(3*((1 - x)^2)) * exp(-(x^2) - ((y + 1)^2)),
		xs, xs'
	)
end

#-----------------------------------------------------------------------------------------
"""
	dejong2( width)

Return a 2D map of a De Jong 2 landscape with dimensions (width, width).
"""
function dejong2( width::Int)
	xs = repeat( range(-10,10,width), 1, width)

	map(
		(x,y) -> sin(2x) / (abs(x)+1) + sin(2y) / (abs(y)+1),
		xs, xs'
	)
end

#-----------------------------------------------------------------------------------------
"""
	diffuse4( heatarray::Matrix{Float64}, diffrate::Float64)

Diffuse heatarray via 4-neighbourhoods with periodic boundary conditions and the given
diffusion rate.
"""
function diffuse4( heatarray::Matrix{Float64}, diffrate=1.0)
	heatarray + 0.25diffrate * (
		circshift(heatarray,( 1,0)) + circshift(heatarray,(0, 1)) +
		circshift(heatarray,(-1,0)) + circshift(heatarray,(0,-1))
		- 4heatarray
	)
end

#-----------------------------------------------------------------------------------------
"""
	diffuse4!( heatarray::Matrix{Float64}, diffrate::Float64)

Diffuse heatarray in place via 4-neighbourhoods with periodic boundary conditions and the given
diffusion rate.
"""
function diffuse4!( heatarray::Matrix{Float64}, diffrate=1.0)
	heatarray[:] = diffuse4(heatarray,diffrate)[:]
	heatarray
end

#-----------------------------------------------------------------------------------------
"""
	diffuse8( heatarray::Matrix{Float64}, diffrate::Float64)

Diffuse heatarray via 8-neighbourhoods with periodic boundary conditions and the given
diffusion rate.
"""
function diffuse8( heatarray::Matrix{Float64}, diffrate=1.0)
	sqrt2 = sqrt(2)						# Some preliminary calculations ...
	oneoveraqrt2 = 1/sqrt2
	sum_contributions = 4 + 2sqrt2

	heatarray + (diffrate/sum_contributions) * (
		circshift(heatarray,( 1,0)) + circshift(heatarray,( 0, 1)) +
		circshift(heatarray,(-1,0)) + circshift(heatarray,( 0,-1)) +
		oneoveraqrt2*(
			circshift(heatarray,( 1,1)) + circshift(heatarray,( 1,-1)) +
			circshift(heatarray,(-1,1)) + circshift(heatarray,(-1,-1))
		) - (sum_contributions*heatarray)
	)
end

#-----------------------------------------------------------------------------------------
"""
	diffuse8!( heatarray::Matrix{Float64}, diffrate::Float64)

Diffuse heatarray in place via 8-neighbourhoods with periodic boundary conditions and the given
diffusion rate.
"""
function diffuse8!( heatarray::Matrix{Float64}, diffrate=1.0)
	heatarray[:] = diffuse8(heatarray,diffrate)[:]
	heatarray
end

#-----------------------------------------------------------------------------------------
"""
	abmplayground( model; alabels, mlabels, kwargs...)

Extends abmexploration to replace the model of the ABMObservable abmobs when user clicks the
"reset model"-Button. This reproduces the behaviour of a NetLogo interface, in which resetting
leads to full reinitialisation of slider settings and agent population.
Original source code of button initialisation at GitHub: shorturl.at/biOX3

Author: Nick Diercksen.
"""
function abmplayground(model, initialiser; kwargs...)
	playgrnd_fig,abmobs = abmexploration(model; kwargs...)

	# Retrieve the Reset button from fig.content[10]:
	reset_btn = nothing
	for element ∈ playgrnd_fig.content
		if element isa Button && element.label[] == "reset\nmodel"
			reset_btn = element
			break
		end
	end
	@assert !isnothing(reset_btn) "Couldn't find the 'Reset-model-button'!"

	"""
	on(reset_btn.clicks) do _
		# Retrieve all keyword arguments from the initialiser function
		# (https://discourse.julialang.org/t/get-the-argument-names-of-an-function/32902/4):
		init_kwargs = Base.kwarg_decl(@which initialiser())

		# Retrieve current model properties
		# (https://stackoverflow.com/questions/38625663/subset-of-dictionary-with-aliases):
		props = abmobs.model.val.properties
		kws = (; Dict([Pair(k, props[k]) for k in (keys(props) ∩ init_kwargs)])...)

		# Replace old model with newly initialised model:
		abmobs.model[] = initialiser(; kws...)
	end
	"""

	(playgrnd_fig,abmobs)
end

end # ... of module AgentTools
