
from django.db import models



class Upload(models.Model):
    upload_id = models.BigIntegerField(primary_key=True)
    project_name = models.CharField(max_length=200)
    upload_file_path = models.TextField(blank=True, null=True)

    class Meta:
        db_table = "Upload"


# class model(models.Model):
#     res_id = models.BigIntegerField(primary_key=True)
#     project_name = models.CharField(max_length=200)
#     model_res = models.TextField(blank=True, null=True)
#
#     class Meta:
#         db_table = "model"


class Sai(models.Model):
    upload_id = models.BigIntegerField(primary_key=True)
    project_name = models.CharField(max_length=200)
    model_res = models.TextField(blank=True, null=True)
    target_column=models.CharField(max_length=200)
    split_percentage=models.IntegerField(default=50)
    # modells=JSONField(null=True, blank=True)
    modells=models.TextField(blank=True, null=True)
    models_path=models.TextField(blank=True,null=True)
    target_variable=models.CharField(max_length=501)
    # mod_data = modells.
    # df_cat = models.TextField(blank=True, null=True)
    # df_int = models.TextField(blank=True, null=True)
    # df = models.TextField(blank=True, null=True)
    # X_test=models.TextField(blank=True, null=True)
    # X_train=models.TextField(blank=True, null=True)
    # y_test=models.TextField(blank=True, null=True)
    # y_train=models.TextField(blank=True, null=True)

    class Meta:
        db_table = "Sai"
class Image(models.Model):
    image_id=models.BigIntegerField(primary_key=True)
    file_path_roc = models.TextField(blank=True, null=True)
    file_path_con = models.TextField(blank=True, null=True)
    class Meta:
        db_table = "Image"

class Segment(models.Model):
    id=models.IntegerField(primary_key=True)
    rule_data=models.TextField()


    class Meta:
        db_table = "Segment"
class Rules(models.Model):
    id=models.IntegerField(primary_key=True)
    rule_data=models.TextField()


    class Meta:
        db_table = "Rules"
