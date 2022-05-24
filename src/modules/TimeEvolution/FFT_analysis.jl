export FFT, FFT_analyze, u_of_t, harmonic_variables_t_lab_frame

"""
$(TYPEDSIGNATURES)

Windowed Fourier transform the timeseries of a `DifferentialEquations.jl` simulation `soln_u` in the rotating frame, with time vector `soln_t` and calculate the quadratures `u,v` and frequencies in the non-rotating frame.

Keyword arguments:
-`window`: windowing function. By default, `window = DSP.Windows.hanning`. See `https://docs.juliadsp.org/stable/windows/` for further options
"""

function FFT(soln_u::Vector{Float64}, soln_t::Vector{Float64}; window = DSP.Windows.hanning)
    w = window(length(soln_t))
    dt = soln_t[2]-soln_t[1]
    
    soln_tuples = Tuple.(zip(soln_u, soln_t))
    
    fft_u =length(soln_t)/sum(w)*[fft(w.*[u[j] for (u,t) in soln_tuples])|> fftshift for j in 1:length(soln_u[1])];
    fft_f = fftfreq(length(soln_t), 1/dt) |> fftshift

    return(fft_u[1] / length(fft_f), fft_f)  # normalize fft_u
end

FFT(soln::OrdinaryDiffEq.ODECompositeSolution; window=DSP.Windows.hanning) = FFT(soln.u, soln.t, window=window)


"""
$(TYPEDSIGNATURES)

Finds peaks in the spectrum and returns corresponding frequency, amplitude and phase.

Keyword arguments:
-`rect`: If `true`, a rectangular window function is assumed. Frequency and phase are therefore corrected according to Huang Dishan, Mechanical Systems and Signal Processing (1995) 9(2), 113–118. 
Else, frequencies and values of spectral peaks are returned with no correction.
"""
function FFT_analyze(fft_u::Vector{ComplexF64}, fft_f; rect::Bool=true)
    # retaining more sigdigits gives more ''spurious'' peaks
    if rect == true
        max_indices, mxval = peakprom(round.( abs.(fft_u), sigdigits=3),minprom =1)
        δf = fft_f[2]-fft_f[1] # frequency spacing
        A1= abs.(fft_u)[max_indices] 
        df = zeros(length(max_indices))
        for i in 1:length(max_indices)
            if abs.(fft_u)[max_indices[i]-1]<abs.(fft_u)[max_indices[i]+1]
                A2= abs.(fft_u)[max_indices[i]+1]
                df[i] = -δf /(A1[i]/A2+1)
            else
                A2= abs.(fft_u)[max_indices[i]-1]
                df[i] = δf /(A1[i]/A2+1)
            end
        end
        return 2*pi*(fft_f[max_indices]-df),  A1.*2, angle.(fft_u)[max_indices]+pi*df/δf
    else
        peak_idx, values = Peaks.findmaxima(abs.(fft_u).^2);
        return 2*pi*fft_f[peak_idx], values, angle.(fft_u)[peak_idx]
    end
end

"""
$(TYPEDSIGNATURES)
Reconstruct the time dependence of `u` or `v` in the rotating frame from a Fourier decomposition in the rotating frame
"""
function harmonic_variables_t_rotating_frame(ω_peaks,A_peaks,ϕ_peaks,times)
    N = length(ω_peaks)
    u = zeros(length(times))
    for m in Int(N/2 - mod(N/2,1))+1:N
        u .+= A_peaks[m]*cos.(ω_peaks[m].*times.+ϕ_peaks[m]);
    end
    return u
end

"""
$(TYPEDSIGNATURES)
Reconstruct the time dependence of `u` or `v` in the lab frame from a Fourier decomposition in the rotating frame
"""
function harmonic_variables_t_lab_frame(ω_rot,ω_peak,A_u_peak,ϕ_u_peak,A_v_peak,ϕ_v_peak)
    ω_nr = [ω_rot-ω_peak, ω_rot+ω_peak]
    u_nr = [-A_u_peak*cos(ϕ_u_peak) + A_v_peak*sin(ϕ_v_peak); -A_u_peak*cos(ϕ_u_peak) - A_v_peak*sin(ϕ_v_peak)]./2
    v_nr = [ A_v_peak*cos(ϕ_v_peak) + A_u_peak*sin(ϕ_u_peak);  A_v_peak*cos(ϕ_v_peak) - A_u_peak*sin(ϕ_u_peak)]./2
    return ω_nr, u_nr, v_nr
end
