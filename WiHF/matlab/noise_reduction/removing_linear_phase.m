function result = removing_linear_phase(csi_data)
% Sanitize CSI by removing linear phase component.
% F(x,y) = Ax^2+2Bxy+Cy^2+2Dx+2Ey+F,
% x_{opt} = BE-CD/(AC-B^2), 
% y_{opt} = BD-AE/(AC-B^2).

    M=3; % Antenna number
    N=30; % Subcarrier number
    freq_delta = 2*312.5e3; % Subcarrier frequency difference
    
    % Unwrap the csi phase
    csi_phase = zeros(1,M*N);
    for jj = 1:M
        if jj == 1
            csi_phase((jj-1)*N+1:jj*N) = unwrap(angle(csi_data((jj-1)*N+1:jj*N)));
        else
            csi_diff = angle(csi_data((jj-1)*N+1:jj*N).*conj(csi_data((jj-2)*N+1:(jj-1)*N)));
            csi_phase((jj-1)*N+1:jj*N) = unwrap(csi_phase((jj-2)*N+1:(jj-1)*N) + csi_diff);
        end
    end
    
    % Linear fitting
    ai = 2*pi*freq_delta*repmat((0:N-1),1,M);
    bi = ones(1, length(csi_phase));
    ci = csi_phase;
    A = ai * ai.';
    B = ai * bi.';
    C = bi * bi.';
    D = ai * ci.';
    E = bi * ci.';
%     F = ci * ci.';
    rho_opt = (B*E-C*D)/((A*C-B^2));
    beta_opt = (B*D-A*E)/((A*C-B^2));
    
    % Sanitization
    csi_phase = csi_phase + 2*pi*freq_delta*repmat((0:N-1), 1, M)*rho_opt + beta_opt;
    result = abs(csi_data) .* exp(1j * csi_phase);
end