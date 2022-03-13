%Raza Ali
%Na 431
%Power estimation
output_power = output.ptos(1).powerInternalMechanics(:,3)*-1/1e3;
powerovertime = cumtrapz(output_power);
totalpower = sum(output.ptos(1).powerInternalMechanics(:,3));
averagepower = totalpower/60;

figure()
plot(output.ptos(1).time(1000:4000,1),output_power(1000:4000,1))
xlabel('Time (s)');
ylabel('Power (W)');
title('Power Generated throughout simulation')
legend('power');

figure()
plot(output.ptos(1).time(1000:4000,1),powerovertime(1000:4000,1))
xlabel('Time (s)');
ylabel('Power (W)');
title('Power Generated Per Minute')
legend('power');

% figure()
% plot(output.ptos(1).time,totalpower)
% plot(output.ptos(1).time,averagepower)
% xlabel('Time (s)');
% ylabel('Power (kW)');
% title('Power Generated throughout simulation')
% legend('power');
