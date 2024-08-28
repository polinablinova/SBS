%clear
%clearvars –global
close all
clc

sigma_s = 33; sigma_p = 70;
xs = 500; xp = 280;
omega_pump = 10; 
omega_0 = 10;
omega_p = 1;
as = 0.01;
apump = 0.01;
dw = 0.00;
v = 1; %damping

xi_max = 1200; T_max = 450;
dxi = 0.5; dt = 0.5;
nxi=floor(xi_max/dxi);
nt =floor(T_max/dt);
ppb = 128; % particles per bucket

Xi=[0:nxi-1]*dxi; TT=[0:nt-1]*dt;


As = zeros(nxi,nt); % seed
Ap = zeros(nxi,nt); % pump
As(:,1) = as*exp(-(Xi-xs).^2/sigma_s^2);
Ap(:,1) = apump*exp(-(Xi-xp).^12/sigma_p^12);
%Ap(:,1) = (apump/2)*(2/pi)*atan(-2*(Xi-xp)) + apump/2;

figure; hold on; plot(Xi,As(:,1)); plot(Xi,Ap(:,1))

% initial phase and velocities 
Phi = zeros(nxi,ppb,nt);
P = zeros(nxi,ppb,nt);

% initial condition
for k=1:nxi
    P(k,:,1) = zeros(1,ppb) + dw;
    Phi(k,:,1) = linspace(-pi,pi,ppb);
end


photons = zeros(1,nt);
photons(1) = 1;
norm = sum(As(:,1).*conj(As(:,1)) + Ap(:,1).*conj(Ap(:,1)));


h=waitbar(0,'Running...');
tt=1;



% constant coefficients for tridiag
ap = zeros(1,nxi-2) - 1/(dxi);
bp = zeros(1, nxi-2) + 1/dt;
cp = zeros(1,nxi-2) + 1/(dxi);

as = zeros(1,nxi-2);
bs = zeros(1, nxi-2) + 1/dt;
cs = zeros(1,nxi-2);

f = zeros(nxi,nt);

for tt=1:nt


    waitbar(tt/nt)
    
    js = zeros(nxi,1);
    jp = zeros(nxi,1);


    w2 = 4 * omega_0^2* As(:,tt) .* Ap(:,tt);
    [m,y]=max(w2);

    
    % USING HUR EQUATIONS
    % Outer loop for xi buckets, RK4
    for k = 1:nxi

        p_b = P(k,:,tt);
        phi_b = Phi(k,:,tt);


        phase1 = sum( exp( +(1i)*phi_b ), "All" ) / ppb;
        phase2 = sum( exp( -(1i)*phi_b ), "All" ) /ppb;
        js(k) = -1i * omega_p^2 * (Ap(k,tt)) .* phase1 / (2*omega_0);
        jp(k) = -1i * omega_p^2 * (As(k,tt)) .* phase2 / (2*omega_pump);

        f(k,tt) = phase2;

        % \omega_b squared array for this bin
        w2 = real(4 * omega_0^2 * As(k,tt) * Ap(k,tt));

        
        % Inner loop for particles in the bucket
        for p = 1:ppb

            k1phi = p_b(p);
            k1p = -w2 * sin( phi_b(p) ) - v*p_b(p);
            
            k2phi = p_b(p) + 0.5 * dt * k1p;
            k2p = -w2 * sin( phi_b(p) + 0.5 * dt * k1phi);
            
            k3phi = p_b(p) + 0.5 * dt * k2p;
            k3p = -w2 * sin( phi_b(p) + 0.5 * dt * k2phi) - v*(p_b(p) + 0.5*k2p);
            
            k4phi = p_b(p) + dt * k3p;
            k4p = -w2 * sin( phi_b(p) + dt * k3phi) - v*(p_b(p) + k3p);

            KPHI = (k1phi + 2*k2phi + 2*k3phi + k4phi);
            KP = (k1p + 2*k2p + 2*k3p + k4p);

    
            Phi(k,p,tt+1) = phi_b(p) + (dt / 6) * KPHI;
            P(k,p,tt+1) = p_b(p) + (dt / 6) * KP;

        end

        

    end



    ds = transpose(js(2:nxi-1) + As(2:nxi-1,tt)/dt);
    dp = transpose(jp(2:nxi-1) + Ap(2:nxi-1,tt)/dt);

    % update envelopes for next time step
    As(2:nxi-1,tt+1)=solve_tridiag(as,bs,cs,ds,nxi-2);
    Ap(2:nxi-1,tt+1)=solve_tridiag(ap,bp,cp,dp,nxi-2);
    %Ap(1:10,tt+1)=apump;
   

    photons(tt) = sum(As(:,tt).*conj(As(:,tt)) + Ap(:,tt).*conj(Ap(:,tt))) / norm;
  

end



close(h)

figure;
nn=0;
for ntt=[floor(nt*[0.2:0.2:0.8])]
nn=nn+1;   
subplot(2,2,nn)
hold on
box on
ax = gca;
pbaspect([1 1 1]);
ax.LineWidth = 1;
plot(Xi,abs(As(:,ntt)), 'r', 'LineWidth',1);
plot(Xi,abs(Ap(:,ntt)), 'k', 'LineWidth',1);
ylim([0,3*apump]);
xlabel('s');
title(['t =',sprintf('%5.1f',ntt*dt)],'fontsize',14,'fontweight','normal');
end
legend("A_s","A_p")


figure;
nn=0;
for ntt=[floor(nt*[0.2:0.2:0.8])]
nn=nn+1;   
subplot(2,2,nn)
hold on
box on
ax = gca;
ax.LineWidth = 1;
plot(Xi,real(1j*f(:,ntt)/omega_0), 'b', 'LineWidth',1);
plot(Xi,imag(1j*f(:,ntt)/omega_0), 'k', 'LineWidth',1);
xlabel('s');ylim([-0.08,0.08])
title(['t =',sprintf('%5.1f',ntt*dt)],'fontsize',18,'fontweight','normal');
end
legend("re(f)","im(f)")


figure;plot(TT,photons)
title('Energy Conservation')
box on
ax = gca;
ax.LineWidth = 1;


figure;
nn=0;
for ntt=[floor(nt*[0.2:0.2:0.8])]
nn=nn+1;   
subplot(2,2,nn)
hold on
box on
ax = gca;
ax.LineWidth = 1;
scatter(Phi(y,:,ntt), P(y,:,ntt));
xlabel('\Phi');
ylabel('P');
title(['t =',sprintf('%5.1f',ntt*dt)],'fontsize',18,'fontweight','normal');
end




figure;hold on;
box on
ax = gca;
ax.LineWidth = 1;
xlim([-pi,pi])
scatter(Phi(y,:,end), P(y,:,end));
legend('initial','end: nz='+string(round(y*dxi)))
xlabel('\Phi');
ylabel('P');
box on
ax = gca;
ax.LineWidth = 1;



ximov = Xi;
apmov = Ap;
asmov = As;
