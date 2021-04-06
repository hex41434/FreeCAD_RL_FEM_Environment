from RLFEM import RLFEM

path = './fc1_Face109_Plus901000.ply'

fc_env = RLFEM(path=path)

fc_env.run()

fc_env.save_result('./')