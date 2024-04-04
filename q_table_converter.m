clear all;close all; clc;

%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 8);

% Specify range and delimiter
opts.DataLines = [2, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["g1", "g2", "g3", "g4", "row", "col", "psi", "action"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Import the data
qtable = readtable("qtable.csv", opts);

%% Convert to output type
qtable = table2array(qtable) + 1;

%% Clear temporary variables
clear opts

for i=1:length(qtable)
    aux = qtable(i,:);
    g1 = aux(1);
    g2 = aux(2);
    g3 = aux(3);
    g4 = aux(4);
    row = aux(5);
    col = aux(6);
    psi = aux(7);
    act = aux(8);
    state2act(g1,g2,g3,g4,row,col,psi) = act;
end

save 'state2act.mat' 'state2act'