function coordinates = update_coordinates_Liu(coordinates,rho,Atdk)
% update coordinates

coordinatesPlus = (coordinates > 0) .* coordinates ; %%% Non-negative coordinates along direction psi_i
coordinatesMinus = coordinatesPlus - coordinates ;   %%% Non-negative coordinates along direction -psi_i

coordinatesPlus = coordinatesPlus + rho * (Atdk-1);
coordinatesMinus = coordinatesMinus + rho * (-Atdk-1);
%coordinatesMinus = coordinatesMinus + rho * (Atdk-1);


coordinatesPlus = (coordinatesPlus > 0) .* coordinatesPlus ;
coordinatesMinus = (coordinatesMinus > 0) .* coordinatesMinus ;

coordinates = coordinatesPlus - coordinatesMinus;  %%% "projection"

return;
