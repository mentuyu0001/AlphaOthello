import pyrev
from pyrev import Position

pos = Position()
print(pos)

print("------------------")
coords = list(pos.get_player_disc_coords())
for coord in coords:
    print(pyrev.coord_to_str(coord))

print()
#print(pos.get_player_disc_coords())
print(pos.do_move_at(pyrev.F5)) # 合法手
print(pos)


#print(pos.get_player_disc_coords())
coords = list(pos.get_player_disc_coords())
print("------------------")
for coord in coords:
    print(pyrev.coord_to_str(coord))
