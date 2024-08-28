clc
%clear all
%close all

% sigma_p = 70; sigma_s = 10; 
% xp = 100; xs = 250; 
% omega_pump = 200; 
% omega_0 = 185;
% omega_p = 5;
% as = 0.005;
% apump = 0.01;
% 
% xi_max = 500; T_max = 140;
% dxi = 0.5; dt = 0.5;

x_max = 1000; T_max = 300;
dx = 0.5; dt = 0.1;
nx=floor(x_max/dx);
nt =floor(T_max/dt);

x=[0:nx-1]*dx; t=[0:nt-1]*dt;

% Initial conditions
sigma_s = 33; sigma_p = 70;
xs = 370; xp = 280;

% w_pump = 10; 
% w_0 = 10;
omega_p = 1;
aa = 0.04;
bb = 0.01;

% Constants
K=1/2;w=10;v=0;do=0;


a_initial = aa  * exp(-(x - xp).^12 / sigma_p^12);
b_initial = bb  * exp(-(x - xs).^2 / sigma_s^2);
%a_initial = (aa/2)*(2/pi)*atan(-2*(x-xp)) + aa/2;
f_initial = zeros(1, nx); % Assuming initial condition for f is zero everywhere

% Define functions for initial conditions
% a_func = @(x) a0 * exp(-(x - xp).^2 / sigma_p^2);
% b_func = @(x) a0 / 3 * exp(-(x - xs).^2 / sigma_s^2);

% Discretize the spatial derivative using central difference
dadx = @(a) [0, diff(a) / dx]; % Central difference for da/dx

% Initialize arrays to store solutions
a = zeros(nt, nx);
b = zeros(nt, nx);
f = zeros(nt, nx);

% Set initial conditions
a(1, :) = a_initial;
b(1, :) = b_initial;
f(1, :) = f_initial;

photons = zeros(1,nt);
photons(1) = 1;
norm = sum(a(1,:).*conj(a(1,:)) + b(1,:).*conj(b(1,:)));

h=waitbar(0,'Running...');
n=1;

% RK4 method implementation
for n = 1:nt-1

    waitbar(n/nt)

    k1_a = dt * ( -2*dadx(a(n, :)) - K * b(n, :) .* f(n, :) );
    k1_b = dt * ( K * a(n, :) .* conj(f(n, :)) );
    k1_f = dt * ( -1i * do * f(n, :) - v * f(n, :) + (w) * a(n, :) .* conj(b(n, :)) );
    
    k2_a = dt * ( -2*dadx(a(n, :) + k1_a/2) - K * (b(n, :) + k1_b/2) .* (f(n, :) + k1_f/2) );
    k2_b = dt * ( K * (a(n, :) + k1_a/2) .* conj(f(n, :) + k1_f/2) );
    k2_f = dt * ( -1i * do * (f(n, :) + k1_f/2) - v * (f(n, :) + k1_f/2) + (w) * (a(n, :) + k1_a/2) .* conj(b(n, :) + k1_b/2) );
    
    k3_a = dt * ( -2*dadx(a(n, :) + k2_a/2) - K * (b(n, :) + k2_b/2) .* (f(n, :) + k2_f/2) );
    k3_b = dt * ( K * (a(n, :) + k2_a/2) .* conj(f(n, :) + k2_f/2) );
    k3_f = dt * ( -1i * do * (f(n, :) + k2_f/2) - v * (f(n, :) + k2_f/2) + (w) * (a(n, :) + k2_a/2) .* conj(b(n, :) + k2_b/2) );
    
    k4_a = dt * ( -2*dadx(a(n, :) + k3_a) - K * (b(n, :) + k3_b) .* (f(n, :) + k3_f) );
    k4_b = dt * ( K * (a(n, :) + k3_a) .* conj(f(n, :) + k3_f) );
    k4_f = dt * ( -1i * do * (f(n, :) + k3_f) - v * (f(n, :) + k3_f) + (w) * (a(n, :) + k3_a) .* conj(b(n, :) + k3_b) );
    
    a(n + 1, :) = a(n, :) + (k1_a + 2*k2_a + 2*k3_a + k4_a) / 6;
    b(n + 1, :) = b(n, :) + (k1_b + 2*k2_b + 2*k3_b + k4_b) / 6;
   % a(n + 1, 1) = aa;
    f(n + 1, :) = f(n, :) + (k1_f + 2*k2_f + 2*k3_f + k4_f) / 6;

    photons(n) = sum(a(n,:).*conj(a(n,:)) + b(n,:).*conj(b(n,:))) / norm;
end

% Plot results
figure;
subplot(3,1,1);
hold on;
plot(x, abs(a(round(nt/3),:) ) );
plot(x, abs(b(round(nt/3),:) ) );
plot(x, abs(f(round(nt/3),:) ) );


subplot(3,1,2);
hold on;
plot(x, abs(a(round(2*nt/3),:) ));
plot(x, abs(b(round(2*nt/3),:) ));
plot(x, abs(f(round(2*nt/3),:) ));

subplot(3,1,3);
hold on;
plot(x, abs(a(round(nt),:) ) );
plot(x, abs(b(round(nt),:) ) );
plot(x, abs(f(round(nt),:) ));

figure;
nn=0;
for ntt=[floor(nt*[0.2:0.2:0.8])]
nn=nn+1;   
subplot(2,2,nn)
hold on
box on
ax = gca;
pbaspect([1 1 1])
ax.LineWidth = 1;
plot(x,abs(a(ntt,:)), 'k', 'LineWidth',1);
plot(x,abs(b(ntt,:)), 'r', 'LineWidth',1);
ylim([0,3*aa]);
xlabel('s');
title(['t =',sprintf('%5.1f',ntt*dt)],'fontsize',14,'fontweight','normal');
end
legend("A_p","A_s")


figure;
nn=0;
for ntt=[floor(nt*[0.2:0.2:0.8])]
nn=nn+1;   
subplot(2,2,nn)
hold on
box on
ax = gca;
ax.LineWidth = 1;
plot(x,real(f(ntt,:)), 'b', 'LineWidth',1);
plot(x,imag(f(ntt,:)), 'k', 'LineWidth',1);
xlabel('s');
title(['t =',sprintf('%5.1f',ntt*dt)],'fontsize',18,'fontweight','normal');
end
legend("re(f)","im(f)")


figure;plot(t,photons)
title('Energy Conservation')
box on
ax = gca;
ax.LineWidth = 1;

close(h)