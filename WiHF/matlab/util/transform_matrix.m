function dist_mtx = transform_matrix(curr_loc, tx_loc, rx_loc)
% DIST_MTX=DISTANCE_MATRIX(CURR_LOC, TX_LOC, RX_LOC) calculated the
% coefficient matrix $A$ in Equation 5 and 6,.
%
% CURR_LOC : Current location of the target. 
% TX_LOC   : Location of the transmitter.
% RX_LOC   : Location of the receivers.
%
% DIST_MTX : Coefficient matrix A.
%

    col_xt = (curr_loc(1) - tx_loc(1)).';
    col_yt = (curr_loc(2) - tx_loc(2)).';
    col_xr = (curr_loc(1) - rx_loc(1,:)).';
    col_yr = (curr_loc(2) - rx_loc(2,:)).';
    
    dist_t = sqrt(col_xt .* col_xt + col_yt .* col_yt);
    dist_r = sqrt(col_xr .* col_xr + col_yr .* col_yr);
    col_x = col_xt ./ dist_t + col_xr ./ dist_r;
    col_y = col_yt ./ dist_t + col_yr ./ dist_r;
    dist_mtx = [col_x, col_y];
end