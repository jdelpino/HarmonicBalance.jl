export add_Hopf!
export Hopf_variables
export replace_Hopf_variable!


"Add a Hopf bifurcation to the system."
function add_Hopf!(eom::DifferentialEquation, Δω::Num)
    for var in get_variables(eom), ω in eom.harmonics[var]
        add_harmonic!(eom, var, ω + Δω)
        add_harmonic!(eom, var, ω - Δω)
    end
end


"Successively adds n Hopf bifurcations at the same frequency Δω"
add_Hopf!(eom::DifferentialEquation; frequency::Num, multiplicity::Int) = [add_Hopf!(eom, frequency) for k in 1:multiplicity]



"""
    Hopf_variables(eom, freq)
"""
Hopf_variables(eom::HarmonicEquation, freq::Num) = filter(x -> any(isequal.(freq, get_all_terms(x.ω))), eom.variables)


function replace_Hopf_variable!(eom::HarmonicEquation, freq::Num)
    pair_to_change = Hopf_variables(eom, freq)[1] # eliminate of the Cartesian variables, it does not matter from which VDP pair
    to_eliminate = pair_to_change.symbols[2]
    rules = Dict(freq => HarmonicBalance.declare_variable(string(freq), first(get_independent_variables(eom))), to_eliminate => Num(0))
    eom.equations = expand_derivatives.(substitute_all(eom.equations, rules))
    eom.parameters = setdiff(eom.parameters, [freq]) # freq is now NOT a parameter
    eom.variables = setdiff(eom.variables, [pair_to_change])
    new_variable = HarmonicBalance.declare_variable(string(freq))
    eom.variables = cat(eom.variables, _VDP_Hopf(pair_to_change, to_eliminate, new_variable), dims=1)
end

function replace_Hopf_variable(eom::HarmonicEquation, freq::Num)
    new_eom = deepcopy(eom)
    replace_Hopf_variable!(new_eom, freq)
    new_eom
end


function _VDP_Hopf(old::HarmonicVariable, eliminated::Num, new_sym::Num)
    new_VDP = deepcopy(old)
    idx = findall(x-> isequal(x, eliminated), old.symbols)[1]
    indep = Num(first(arguments(eliminated.val)))
    new_VDP.symbols[idx] = HarmonicBalance.declare_variable(string(new_sym), indep)
    new_VDP.types[idx] = "Hopf"
    new_VDP.names[new_sym] = string(new_sym)
    delete!(new_VDP.names, Num(var_name(eliminated)))
    return new_VDP
end

