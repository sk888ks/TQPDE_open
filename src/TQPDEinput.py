try:
    from TQPDEinput_temp import *
    # directry path
    dir_QPDE_path = get_dir_QPDE_path()

    # IBM Quantum API token and information for your hub
    IBM_token = get_token()
    hub, group, project = get_hub_group_project()

    # If you can use FireOpal, please replace api key and organization
    FireOpal_api_key = get_api_key()
    organization_slug = get_organization_slug()

except ImportError:

    # directry path
    dir_QPDE_path = "/xxxx/TQDE_open" 

    # IBM Quantum API token and information for your hub
    IBM_token = "xxxx"
    hub = "xxxx"
    group = "xxxx"
    project = "xxxx"

    # If you can use FireOpal, please replace api key and organization
    FireOpal_api_key = "xxxx"
    organization_slug="xxxx"

    def get_dir_QPDE_path():
        return dir_QPDE_path

    def get_token():
        return IBM_token

    def get_hub_group_project():
        return hub, group, project

    def get_api_key():
        return FireOpal_api_key

    def get_organization_slug():
        return organization_slug

