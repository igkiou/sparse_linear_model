fprintf('Compiling all mex functions ...\n');
cd mexfunctions
try
mex cdist.c
catch
fprintf('Could not compile cdist.c\n');
end;
try
mex findimps3Dac.c
catch
fprintf('Could not compile findimpDac.c\n');
end;
try 
mex mink.c
catch
fprintf('Could not compile mink.c\n');
end;
try 
mex SOD.c 
catch
fprintf('Could not compile SOD.c \n');
end;

try 
mex sumiflessh2.c
catch
fprintf('Could not compile ...\n');
end;

try 
mex count.c
catch
fprintf('Could not compile sumiflessh2.\n');
end;

try 
mex findlessh.c   
catch
fprintf('Could not compile findlessh.c \n');
end;

try 
mex sd.c  
catch
fprintf('Could not compile sd.c \n');
end;

try 
mex SODW.c
catch
fprintf('Could not compile SODW.c  \n');
end;

try 
mex sumiflessv2.c
catch
fprintf('Could not compile sumiflessv2.c  \n');
end;
addpath(pwd);
cd ..

cd mtrees
try 
mex buildmtreec.cpp
catch
fprintf('Could not compile buildmtreec.cpp\n');
end;    

try
    mex findknnmtree.cpp  
catch
    fprintf('Could not execute mex findknnmtree.cpp  ')    
end;    

try
    mex findNimtree.cpp  
catch
    fprintf('Could not execute mex findNimtree.cpp  ')    
end;    
addpath(pwd);
cd ..


%%mex distance.c  %% de-comment this line for maximum speed on 64 processors
fprintf('done\n\n');





