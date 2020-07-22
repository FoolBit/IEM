function color = changeSaturation(rgb, s)

hsv = rgb2hsv(rgb);
hsv(2) = s;
color = hsv2rgb(hsv);

end

