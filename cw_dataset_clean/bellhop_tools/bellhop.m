function bellhop( filename )

% run the BELLHOP program
%
% usage: bellhop( filename )
% where filename is the environmental file

runbellhop = which( 'bellhop.exe' );
eval( [ '! "' runbellhop '" ' filename ] );
