clear all;

N=20;   %Number of sections in the traffic model

K=360*2;    %Time horizon of the traffic model

T=1/3600*10;    %Set time interval to 10 s


Ir=zeros(N,1);    % Initialize a vector that has 1 at positions where there is an on-ramp
Ir(5)=1;
Ir(10)=1;
Ir(12)=1;

 
L=zeros(N,1);    % Initialize a vector that stores the length of each section that is 500m
for i=1:N
   L(i,1)=500/1000;
end


rho_max=zeros(N,1);    % Initialize a vector that stores the maximum density for each section
for i=1:N
    rho_max(i,1)=600;
end


vf=zeros(N,1);    % Initialize a vector that stores the free-flow speed for each section
for i=1:N
    vf(i,1)=100;
end


w=zeros(N,1);     % Initialize a vector that stores the capacity drop due to an incident for each section
for i=1:N
    w(i,1)=27;
end


pr=zeros(N,1);     % Initialize a vector that stores the probability of a vehicle using an onramp for each section
for i=1:N
    pr(i,1)=0.1;
end


rho_iniz=zeros(N,1);    % Initialize a vector that stores the initial density for each section
fi_iniz=zeros(N,1);    % Initialize a vector that stores the initial flow rate in each section of the road network
for i=1:N
    rho_iniz(i,1)=60;
    fi_iniz(i,1)=rho_iniz(i,1)*100;
end


rho_sez0=zeros(K,1);    % Initialize a vector that stores the initial density for each section at each time step
fi_sez0=zeros(K,1);    % Initialize a vector that stores the flow rate demand in each time step 
for k=1:K
    rho_sez0(k,1)=60;
    fi_sez0(k,1)=rho_sez0(k,1)*100;
end


rho_sezfin=zeros(K,1);     % Initialize a vector that stores the final density for each section at each time step
fi_sezfin=zeros(K,1);     % Initialize a vector that stores the flow rate demand in each time step
for k=1:K
    rho_sezfin(k,1)=80;
    fi_sezfin(k,1)=rho_sezfin(k,1)*100;
end


dem=zeros(N,K);    % Initialize a matrix that stores the demand for each section at each time step
r=zeros(N,K);    % Initialize a matrix that stores the actual flow for each section at each time step
for k=1:K
    dem(5,k)=100;
    dem(10,k)=100;
    dem(12,k)=50;
end


rho=zeros(N,K+1);    % Initialize a matrix that stores the density for each section at each time step
ql=zeros(N,K+1);    % Initialize a matrix that stores the queue length for each section at each time step
fi=zeros(N,K+1);    % Initialize a matrix that stores the flow for each section at each time step


for i=1:N-1
     rho(i,1)=rho_iniz(i)+T/L(i)*(fi_iniz(i)-fi_iniz(i+1));    % Compute the initial density for each section using the initial flow
end

rho(N,1)=rho_iniz(N)+T/L(N)*(fi_iniz(N)-vf(N)*rho_iniz(N));    % Compute the density for the last section


for j = 1:K
    fi(1,j) = min([fi_sez0(j), (1-pr(i,1)) * w(i,1) * (rho_max(i,1) - rho(i,j))]);    % Calculate flow into the first section of road segment
        
    for i=1: N-1
        fi(i+1,j) = min([vf(i,1)*rho(i,j),(1-pr(i+1,1))*w(i+1,1)*(rho_max(i+1,1)-rho(i+1,j))]);    % Calculate flow into remaining sections of road segment
    end

    % Calculate outflow from each section of road segment
    for i = 1:N
        r(i,j) = min([dem(i,j) + (ql(i,j)/T), pr(i,1) * w(i,1) * (rho_max(i,1) - rho(i,j))]);     % Calculate the maximum flow that can leave the section of road segment
        
        if i<N
            rho(i, j+1) = rho(i,j) + (T/L(i,1)) * (fi(i,j) + r(i,j) - fi(i+1, j));    % Calculate the change in density for each section of road segment except the last one
        else 
            fi_sezfin(j) = min([vf(N) * rho(N, j), fi_sezfin(j)]);    % Calculate the minimum flow into the last section of road segment
            
            rho(i, j+1) = rho(i,j) + (T/L(i)) * (fi(i,j) - fi_sezfin(j));    % Calculate the change in density for the last section of road segment
        end
        
        ql(i, j+1) = ql(i,j) + T * (dem(i,j) - r(i,j));    % Calculate the change in queued vehicles for each section of road segment
    end
end

mesh(rho);   % Plot the density matrix as a mesh plot


rc=zeros(N,K+1);    % Initialize a matrix that stores the values of the ramp metering rate control for each time step and each section
kr = 3; 
rho_star = 100;


for k=1:K
    for i=1:N
        if(Ir(i)==1)   % Check if there is an on-ramp in section i
            if(k==1)
                rc(i,k) = kr*(rho_star - rho(i,k));   % Calculate the ramp control input based on the deviation from the desired density
            else
                rc(i,k) = r(i,k-1)+kr*(rho_star - rho(i,k));  % Calculate the ramp control input based on the deviation from the desired density and the previous ramp control input
            end
    
        end
    end
end


