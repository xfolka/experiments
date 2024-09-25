import omero
import ezomero
from omero.gateway import BlitzGateway
from pprint import pprint

def print_obj(obj, indent=0):
    """
    Helper method to display info about OMERO objects.
    Not all objects will have a "name" or owner field.
    """
    print("""%s%s:%s  Name:"%s" (owner=%s)""" % (
        " " * indent,
        obj.OMERO_CLASS,
        obj.getId(),
        obj.getName(),
        obj.getOwnerOmeName()))


uname = "xfolka"
upass = "tmp123tmp"
uhost = "omero.cci.sahlgrenska.gu.se"
uport = 4064
conn = BlitzGateway(uname,upass, host=uhost, port=uport, secure=True)
conn.connect()

user = conn.getUser()
print("Current user:")
print("   ID:", user.getId())
print("   Username:", user.getName())
print("   Full Name:", user.getFullName())

iids = ezomero.get_image_ids(conn, dataset=168)

imageid = 285

mid = ezomero.get_map_annotation_ids(conn, "Image",284)

ann = ezomero.get_map_annotation(conn, mid[0])

ns = 'jax.org/jax/example/namespace'
annotations = {'species': 'human',
                'occupation': 'time traveler',
                'first name': 'Kyle',
                'surname': 'Reese'}

nmid = ezomero.post_map_annotation(conn,"Image", imageid,{"key": "a2", "value": "testar"},"")


#tags = conn.getObjects("Image")
#for ta in tags:
#    print_obj(ta)




conn.close()
