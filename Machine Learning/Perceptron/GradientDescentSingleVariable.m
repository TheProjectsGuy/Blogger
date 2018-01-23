## Copyright (C) 2018 Stark
## 
## This program is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

## -*- teXinfo -*- 
## @deftypefn {} {@var{retval} =} GradientDescentSingleVariable (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Stark <stark@Jarvis>
## Created: 2018-01-22

function [newWeights, newCost] = GradientDescentSingleVariable (x, y, alpha, weights)
  % Apply single variable gradient descent to your data set, just once
  m = length(y);  % Length of dataset
  X = [ones(m,1) x];
  deviation = (X*weights - y);
  delta = sum([deviation, deviation].*X)';
  newWeights = weights - (alpha/m) * delta;
  newCost = CostFunction(x, y, newWeights);
end
