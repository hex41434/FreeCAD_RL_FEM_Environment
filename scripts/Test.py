from RLFEM import RLFEM

path = ''

fc_env = RLFEM(path=path)

fc_env.run()

fc_env.save_result('./')