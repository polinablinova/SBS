function [x]=solve_tridiag(a,b,c,d,n)



%      |b1 c1       0    | |x1|  |d1|
%      |a2 b2 c2         | |x2|  |d2| 
%      |   a3 b3 .       | |. |= |. |
%      |      .  .  c_n-1| |. |  |. |
%      |0        an bn   | |xn|  |dn|

% TIANHONG
% cp=zeros(1,n);
% dp=zeros(1,n);
% 
% cp(1)=c(1)/b(1);
% dp(1)=d(1)/b(1);
% x=zeros(n,1);
% 
% % solve for vectors c-prime and d-prime
%          for i = 2:n
%            m = b(i)-cp(i-1)*a(i);
%            cp(i) = c(i)/m;
%            dp(i) = (d(i)-dp(i-1)*a(i))/m;
%          end
% % initialize x
%          x(n) = dp(n);
% % solve for x from the vectors c-prime and d-prime
%        for i = n-1:-1:1
%           x(i) = dp(i)-cp(i)*x(i+1);
%        end
% 
% 
% end


c(1)=c(1)/b(1);
d(1)=d(1)/b(1);
x=zeros(n,1);

% solve for vectors c-prime and d-prime
         for i = 2:n
           m = b(i)-c(i-1)*a(i);
           c(i) = c(i)/m;
           d(i) = (d(i)-d(i-1)*a(i))/m;
         end
% initialize x
         
         x(n) = d(n);
% solve for x from the vectors c-prime and d-prime
       for i = n-1:-1:1
          x(i) = d(i)-c(i)*x(i+1);
       end


end