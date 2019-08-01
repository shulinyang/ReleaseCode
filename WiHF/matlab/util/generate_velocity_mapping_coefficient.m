function coefficient = generate_velocity_mapping_coefficient(curr_pos, Tx_pos, Rx_pos, n_receivers)
% coefficient = generate_velocity_mapping_coefficient(curr_pos, Tx_pos, Rx_pos, n_receivers) 
% calculated the coefficient matrix $A$ in Equation 5 and 6,.
%
% CURR_POS : Current location of the target. 
% TX_POS   : Location of the transmitter.
% RX_POS   : Location of the receivers.
%
% COEFFIECIENT : Coefficient matrix A.
%
if n_receivers > size(Rx_pos,1)
    error('Error Rx Count!')
end
coefficient = zeros(n_receivers,2);

for ii = 1:n_receivers
    dis_torso_tx = sqrt((curr_pos-Tx_pos) * (curr_pos-Tx_pos)');
    dis_torso_rx = sqrt((curr_pos-Rx_pos(ii,:)) * (curr_pos-Rx_pos(ii,:))');
    coefficient(ii,:) = (curr_pos - Tx_pos)/dis_torso_tx + ...
        (curr_pos - Rx_pos(ii,:))/dis_torso_rx;
end
end